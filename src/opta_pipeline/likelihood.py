"""ψ/φ likelihood track-and-stack (Phase 1 of the stack rework).

Reformulates blind shift-and-add as trajectory summation of per-frame
matched-filter likelihood images, following the KBMOD formulation
(Whidden et al. 2019; Smotherman et al. 2021).

For frame k with pixel data f_k, per-pixel noise variance V_k, and a
unit-sum PSF kernel P, two images carry everything a point-source
maximum-likelihood fit needs:

    ψ_k = correlate(f_k / V_k, P)
    φ_k = correlate(1 / V_k, P²)

For a linear trajectory x(t) = x₀ + v·t the co-added statistics are sums
along the trajectory:

    Ψ(x₀; v) = Σ_k ψ_k(x₀ + v·t_k)        Φ(x₀; v) = Σ_k φ_k(x₀ + v·t_k)

    SNR(x₀; v)  = Ψ / √Φ     — unit-variance Gaussian under pure noise
    flux(x₀; v) = Ψ / Φ      — ML point-source flux estimate

What this fixes relative to :class:`~opta_pipeline.stack.Stacker`
(assessment findings F3 / F4-point / F6):

- **Matched filter**: the optimal point-source statistic, not peak pixel.
- **Coverage**: out-of-frame or masked pixels enter with V = ∞ ⇒ φ = 0,
  so edges and partial visibility are correctly down-weighted and
  different velocity hypotheses become statistically comparable.
- **Inverse-variance frame weighting** falls out of the formulation.
- **Calibrated SNR**: under pure noise SNR(x) has mean 0 and variance 1
  per pixel, so detection thresholds follow from trials statistics alone
  (see "Detection threshold" below and scripts/stacking_far_mc.py).
- **No interpolation in the search**: trajectories are sampled at integer
  pixel offsets (round(v·t_k), error ≤ 0.5 px ≪ PSF).  Sub-pixel
  registration is reserved for the winning hypothesis downstream.

Detection threshold — Poisson-aware tail bound (contract since 2026-07-28)
--------------------------------------------------------------------------
The trial statistic is a *weighted sum of low-count terms*, not a
Gaussian.  Writing the centred data sum behind one SNR-map pixel as

    Y = Σ_i a_i (f_i − E f_i),   a_i = P(p_i − x_k) / V_k ≥ 0

(one term per frame k × PSF-footprint pixel p_i), each calibrated pixel
value decomposes as f = s·N + G with N ~ Poisson (photo-electron /
quantisation events of elementary size s, in the frame's electron units)
and G Gaussian (read noise, calibration residuals).  Its log-MGF is

    log E e^{θY} = Σ_i μ_i (e^{a_i s θ} − 1 − a_i s θ)
                   + θ²/2 · Σ_i a_i² σ_{G,i}²
                 ≤ (V/b²)(e^{bθ} − 1 − bθ),

with V = Σ a_i² V_i the total variance (= Φ when V_i are the assumed
per-frame variances) and b = s·max_i a_i the largest single-event
influence, using that (eˣ−1−x)/x² is non-decreasing (Poisson terms) and
eˣ−1−x ≥ x²/2 (Gaussian terms).  Chernoff optimisation of that envelope
is Bennett's inequality; its standard relaxation is **Bernstein's**:

    P(Y/√V ≥ u) ≤ exp( −u² / (2 (1 + ε u / 3)) ),   ε = b/√V.

Bernstein is chosen over Bennett because it inverts in closed form for
the threshold and errs on the conservative side (its tail dominates
Bennett's, so the false-alarm guarantee is preserved).  Requiring the
expected number of exceedances over the N = N_pix·N_hyp trials budget to
stay ≤ 1 — the same design point the old Gaussian extreme-value bound
√(2 ln N) encodes — and solving N·exp(−u²/(2(1+εu/3))) = 1 with
L = ln N gives the shipped threshold (:func:`poisson_tail_threshold`):

    u*(ε) = Lε/3 + √((Lε/3)² + 2L)   [+ far_margin_sigma on top].

ε → 0 recovers the Gaussian contract exactly; low counts (small λ = V·…)
or coarse quantisation raise it.  Honesty of the two inputs:

* **Variance proxy**: the per-frame ``noise_rms²`` the pipeline itself
  normalises the statistic with (mad_std-based, clamped ≥ 1 e⁻).  The
  bound is valid whenever the assumed variance ≥ the true one; the clamp
  errs in the safe direction at low counts.  The known MAD-discreteness
  bias of the σ *estimator* on integer-quantised data (−11 % at λ = 25)
  is deliberately NOT "corrected" here (no 3σ-clip factor — see the
  repo-wide caution): it is a property of the σ estimate, not of this
  tail bound, and remains a documented residual.
* **Bound scale b** (:func:`bound_scale_epsilon`): b = P_max·s/V with
  s ≥ 1 e⁻ — the physical photo-electron quantum of the calibrated
  electron-unit contract — raised to the recorded-value ladder step when
  the frames are quantised coarser (8-bit video widened to uint16,
  gain > 1 e⁻/ADU), measured from consecutive-frame differences by
  :func:`quantisation_step_e`.  Both are properties of the data/frames,
  never of any target scene (no inverse crime: nothing here is fitted to
  a wanted detection outcome).  The representative Φ is the
  full-coverage value Σ_k ‖P‖²/V_k; border pixels with partial coverage
  have locally larger ε, accepted because they carry proportionally less
  of the trials mass.

Measured effect (blank SMOKE scenes, seeds 0–99, blessed pins): the
Gaussian gate u* = 5.96 was exceeded by 9/100 blank low-count Poisson
scenes (Gaussian control 1/100) — the shipped Bernstein gate covers all
measured exceedances (max honest peak 6.28).  See
tests/test_stack_far.py and the 2026-07-28 SYSTEMS.md decision-log entry
for the before/after numbers with their generating commands.

Within-exposure trailing (F4) is handled by an *opt-in* velocity-matched
streak kernel (:func:`streak_psf_kernel` / :func:`matched_streak_kernel`):
a source moving at |v| px/s smears into a trail of length |v|·t_exp within
each frame, and matched-filtering that trail with the round point PSF loses
SNR.  The blind velocity *search* still uses the round kernel (one kernel
shared across all hypotheses keeps the grid cheap); the streak kernel is
applied where the velocity is already known — the coarse-to-fine final
gate, the prior/single-stage winner re-score, and forced photometry —
gated on the trail clearing ``streak_min_length_px`` so slow movers, short
exposures and wide fields are byte-identical to the round-kernel result.
The velocity-grid sampling criterion of ``Stacker.from_prior`` applies
unchanged.

Usage
-----
    from opta_pipeline.likelihood import PsiPhiStacker

    stacker = PsiPhiStacker.from_config(config)  # sigma from config / synth PSF
    result = stacker.stack(frames, noise_rms, frame_times_s)
    # result.snr_map is N(0,1)-calibrated: threshold by trials statistics
    candidates = np.argwhere(result.snr_map >= threshold)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import correlate as ndimage_correlate
from scipy.special import erf

from opta_pipeline.stack import symmetric_velocity_axis

if TYPE_CHECKING:
    from typing_extensions import TypeIs

__all__ = [
    "DEFAULT_PSF_SIGMA_PX",
    "SYNTH_PSF_FWHM_PX",
    "ForcedMeasurement",
    "LikelihoodStackResult",
    "PsiPhiStacker",
    "bound_scale_epsilon",
    "forced_measurement",
    "gaussian_psf_kernel",
    "make_psi_phi",
    "matched_streak_kernel",
    "median_subtraction_safe",
    "poisson_tail_threshold",
    "quantisation_step_e",
    "streak_length_px",
    "streak_psf_kernel",
    "temporal_median_subtract",
]

_PHI_EPS = 1e-12

# Scalar-vs-per-frame ``noise_rms`` discrimination, shared by every entry point
# that accepts either form (``make_psi_phi`` here, ``blind_coarse_fine_search``
# in coarse_fine).  NumPy scalars are the usual callers — ``float(arr.std())``
# yields np.float64, a calibrate/detect path can hand over np.float32, and an
# integer noise floor arrives as np.int64 — and each one that a predicate misses
# falls through to the sequence branch, where ``noise_rms[i]`` raises IndexError
# on a 0-d value.  One predicate so the entry points cannot drift apart again.
_SCALAR_NOISE_TYPES = (int, float, np.floating, np.integer)


def _is_scalar_noise_rms(noise_rms: object) -> TypeIs[float]:
    """True iff ``noise_rms`` is a single σ shared by all frames.

    0-d arrays count: ``np.mean(...)`` / ``arr[()]`` / ``np.asarray(3.0)`` are
    scalars that are not NumPy *scalar types*, and they fail the sequence
    branch exactly like the missing scalar types did.

    The ``TypeIs[float]`` return narrows both branches for the type checker
    (True → scalar, False → the sequence arm of the union); the runtime
    check is deliberately wider than ``float`` (NumPy scalars, 0-d arrays) —
    everything it accepts is consumed via ``float(...)``.
    """
    if isinstance(noise_rms, np.ndarray):
        return noise_rms.ndim == 0
    return isinstance(noise_rms, _SCALAR_NOISE_TYPES)


# Gaussian FWHM = 2.355 σ — the repo-wide conversion constant (coarse_fine.py,
# sensitivity.py use it identically).
_FWHM_PER_SIGMA = 2.355

# Synthetic-frame PSF the blind matched filter runs against.  opta_pipeline.synth
# renders point sources at ``psf_fwhm_px = 2.0`` (synth/render.py,
# synth/__init__.py — used verbatim by fig_22/fig_32 and the acceptance tests),
# i.e. a Gaussian of σ = FWHM / 2.355 ≈ 0.85 px.  The matched filter MUST use
# that σ: a wider kernel (the former 1.5 px default — a σ↔FWHM confusion, 1.5
# being plausible only as a *FWHM*) forfeits ~14 % matched-filter SNR
# (2σ₁σ₂/(σ₁²+σ₂²) = 0.857 at σ_filter = 1.5 vs σ_synth = 0.85).  This is the
# single fallback default for every ψ/φ entry point; the *production* value is
# ``pipeline_defaults.yaml::stacking.psf_sigma_px`` (read via PipelineConfig),
# kept equal to this so the two cannot drift.  Follow-up (TODO): wire σ straight
# from the synth ``psf_fwhm_px`` / a config PSF model so there is exactly one
# literal instead of two kept-in-sync ones.
SYNTH_PSF_FWHM_PX = 2.0
DEFAULT_PSF_SIGMA_PX = SYNTH_PSF_FWHM_PX / _FWHM_PER_SIGMA  # ≈ 0.8493 px

# Minimum total motion, in PSF-FWHM units, for temporal-median subtraction
# to leave a mover's flux mostly intact.  A source spends ~footprint/(v·T)
# of the frames on any one pixel; with the Gaussian footprint ≈ ±2σ ≈
# 1.7·FWHM the median is contaminated (>50% dwell) below v·T ≈ 3.4·FWHM.
# Measured retention on SENSOR_SMALL synth scenes (pedestal-corrected SNR
# at the true velocity, subtraction on/off): 0.48 at v·T/FWHM = 1.4, 0.68
# at 2.8, 0.76 at 5.7, 0.88 at 11.3, ≥0.93 at 33.
_MEDIAN_SAFE_MOTION_FWHM = 4.0


def median_subtraction_safe(
    v_max_px_s: float, t_span_s: float, psf_sigma_px: float
) -> bool:
    """True when the *fastest searched* motion clears the median blind spot.

    Temporal-median subtraction assumes a mover occupies any pixel in only
    a few frames.  Sources with total motion v·T ≲ 4 PSF-FWHM are treated
    as static scene and subtracted along with it (see
    ``_MEDIAN_SAFE_MOTION_FWHM``).  When even the fastest velocity in the
    search band fails that bound, subtraction can only destroy in-scope
    signal — the caller should skip it and rely on catalog-star masking.
    Above the bound subtraction is applied; velocities near the bound are
    still partially suppressed (documented sensitivity trade).
    """
    psf_fwhm = 2.355 * psf_sigma_px
    return v_max_px_s * t_span_s >= _MEDIAN_SAFE_MOTION_FWHM * psf_fwhm


def gaussian_psf_kernel(sigma_px: float, truncate: float = 4.0) -> np.ndarray:
    """Return a unit-sum 2-D Gaussian PSF kernel.

    Parameters
    ----------
    sigma_px : float
        Gaussian σ in pixels (FWHM = 2.355 σ).
    truncate : float
        Kernel half-extent in units of σ.
    """
    if sigma_px <= 0.0:
        raise ValueError(f"sigma_px must be > 0, got {sigma_px}")
    half = max(1, int(math.ceil(truncate * sigma_px)))
    ax = np.arange(-half, half + 1, dtype=np.float64)
    g1 = np.exp(-(ax**2) / (2.0 * sigma_px**2))
    kernel = np.outer(g1, g1)
    return kernel / kernel.sum()


def poisson_tail_threshold(
    n_trials: int, bound_scale: float, far_margin_sigma: float = 0.0
) -> float:
    """Bernstein detection threshold u* for the calibrated SNR map.

    Solves N·exp(−u²/(2(1 + εu/3))) = 1 for u — the Poisson-aware
    replacement of the Gaussian extreme-value bound √(2 ln N) (module
    docstring, "Detection threshold").  With L = ln(N) and a = Lε/3:

        u* = a + √(a² + 2L)   (+ ``far_margin_sigma``)

    ``bound_scale`` is ε = b/√Φ from :func:`bound_scale_epsilon`; ε = 0
    reproduces the Gaussian contract exactly.  Monotone in both N and ε.
    """
    n = max(int(n_trials), 2)
    log_trials = math.log(n)
    a = log_trials * max(0.0, float(bound_scale)) / 3.0
    return a + math.sqrt(a * a + 2.0 * log_trials) + float(far_margin_sigma)


def bound_scale_epsilon(
    psf_kernel: np.ndarray,
    noise_rms: float | Sequence[float],
    n_frames: int | None = None,
    quant_step_e: float = 1.0,
) -> float:
    """Bernstein bound scale ε = b/√Φ of the trajectory-summed statistic.

    b = max_k(P_max · s / σ_k²) is the largest single-event influence on
    Ψ (one elementary count of size ``quant_step_e`` electrons landing on
    the peak kernel weight of the noisiest-weighted frame); the
    representative statistical weight is the full-coverage
    Φ = Σ_k ‖P‖² / σ_k².  For scalar σ over N frames this reduces to

        ε = (P_max/‖P‖₂) · s / (σ √N)

    — vanishing with √N (CLT) and with rising counts (s/σ = 1/√λ for
    Poisson λ in electron units), inflating exactly where the measured
    under-coverage lives: few-frame, low-count, coarsely-quantised
    windows.  Uses the *assumed* per-frame variances (the same ones the
    SNR statistic is normalised with) — see the module docstring for why
    that direction is the honest one.

    Parameters
    ----------
    psf_kernel : np.ndarray
        The unit-sum matched-filter kernel actually applied.
    noise_rms : float | sequence of float
        Per-frame σ (scalar = shared); must match the ψ/φ build.
    n_frames : int | None
        Required when ``noise_rms`` is scalar; ignored otherwise.
    quant_step_e : float
        Elementary count size s in the frames' (electron) units —
        ``max(1 photo-electron, recorded quantisation ladder)`` from
        :func:`quantisation_step_e`.
    """
    if _is_scalar_noise_rms(noise_rms):
        if n_frames is None or n_frames < 1:
            raise ValueError("n_frames is required for scalar noise_rms")
        sigmas = [float(noise_rms)] * int(n_frames)
    else:
        sigmas = [float(s) for s in noise_rms]
    if not sigmas or any(s <= 0.0 for s in sigmas):
        raise ValueError("noise_rms entries must be > 0")
    if quant_step_e <= 0.0:
        raise ValueError(f"quant_step_e must be > 0, got {quant_step_e}")
    p_max = float(np.max(psf_kernel))
    p_sq = float(np.sum(psf_kernel * psf_kernel))
    b = max(p_max * quant_step_e / (s * s) for s in sigmas)
    phi_full = sum(p_sq / (s * s) for s in sigmas)
    return b / math.sqrt(phi_full)


def quantisation_step_e(
    frames: Sequence[np.ndarray], floor_e: float = 1.0
) -> float:
    """Elementary jump size s (electrons) of the recorded value ladder.

    Estimated from consecutive-frame differences (same pixel, adjacent
    epochs — the same flat/dark divisor, so a coarse quantisation grid
    survives calibration exactly, up to the smooth background mesh's
    slow drift):

    * a resolvable ladder keeps a large fraction of pixel pairs on the
      same rung (differences ≈ 0 at a tolerance well below the dominant
      jump), while continuous-valued noise puts ≲ 7 % of |Δ| below
      5 % of its median — the 10 % tie-fraction gate separates the two;
    * when a ladder is detected, s is the smallest common positive jump
      (5th percentile of the above-tolerance differences).

    Floored at ``floor_e`` = 1 e⁻ — the physical photo-electron quantum
    of the calibrated electron-unit contract: read-noise smoothing can
    hide the ladder (then the Gaussian part genuinely lightens the tail
    at that scale and the floor is the honest Poisson jump), but no
    recorded ladder can make the elementary event *smaller* than one
    photo-electron.  Known residual: frames NOT in electron units with
    sub-electron effective gain would overstate s slightly — the
    conservative direction.  Deterministic (stride subsampling, no RNG);
    a property of the frames, never of a target scene.
    """
    if len(frames) < 2:
        return floor_e
    # Cap the sample deterministically: ≤ 8 consecutive pairs, ≤ 200k
    # pixels per pair (stride subsample) — plenty for a ladder census.
    max_pairs = 8
    max_px = 200_000
    diffs: list[np.ndarray] = []
    step_pairs = max(1, (len(frames) - 1) // max_pairs)
    for i in range(0, len(frames) - 1, step_pairs):
        a = np.asarray(frames[i], dtype=np.float64).ravel()
        b = np.asarray(frames[i + 1], dtype=np.float64).ravel()
        stride = max(1, a.size // max_px)
        diffs.append(np.abs(a[::stride] - b[::stride]))
        if len(diffs) >= max_pairs:
            break
    d = np.concatenate(diffs)
    pos = d[d > 0.0]
    if pos.size == 0:  # perfectly constant frames: no ladder information
        return floor_e
    dominant = float(np.percentile(pos, 50.0))
    if dominant <= 0.0:
        return floor_e
    tol = 0.05 * dominant
    tie_frac = float(np.mean(d <= tol))
    if tie_frac < 0.10:  # continuous-valued: no resolvable ladder
        return floor_e
    jumps = pos[pos > tol]
    if jumps.size == 0:
        return floor_e
    return max(floor_e, float(np.percentile(jumps, 5.0)))


def streak_length_px(v_px_s: float, exposure_s: float) -> float:
    """Within-exposure trail length (px) of a source moving at ``v_px_s``.

    A source drifting at |v| pixels/second during an exposure of
    ``exposure_s`` seconds smears into a trail of length ``|v|·exposure``.
    The result is in *pixels*, so it already folds in the optics: the same
    angular rate gives a longer pixel trail on a narrow field (small
    arcsec/px) than on a wide one, because ``v_px_s`` = v_angular / plate
    scale.  Returns 0 when either factor is non-positive.
    """
    if v_px_s <= 0.0 or exposure_s <= 0.0:
        return 0.0
    return float(v_px_s) * float(exposure_s)


def streak_psf_kernel(
    sigma_px: float,
    length_px: float,
    angle_rad: float,
    truncate: float = 4.0,
) -> np.ndarray:
    """Unit-sum motion-blurred ("streak") matched-filter kernel (F4).

    Models a point source of Gaussian PSF σ that drifts uniformly over a
    trail of length ``length_px`` at position angle ``angle_rad`` during the
    exposure — the correct point-spread for a source that moves *within* a
    single frame (within-exposure trailing).  It is the round PSF convolved
    with a uniform line segment, evaluated in closed form: for the
    along-track coordinate ``a`` and cross-track coordinate ``c`` of each
    kernel pixel,

        K(a, c) ∝ exp(-c²/2σ²) · [Φ((a + L/2)/σ) − Φ((a − L/2)/σ)]

    with Φ the standard-normal CDF and L the trail length.  As ``length_px``
    → 0 this reduces exactly to :func:`gaussian_psf_kernel` (verified in the
    tests), so it is a strict generalization.  The kernel is normalized to
    unit sum, so the ψ/φ statistic stays calibrated (Ψ/√Φ ~ N(0,1) under
    noise for *any* fixed kernel; unit sum makes Ψ/Φ an unbiased flux).

    Parameters
    ----------
    sigma_px : float
        Gaussian PSF σ in pixels (FWHM = 2.355 σ).
    length_px : float
        Trail length L in pixels (:func:`streak_length_px`).  ``<= 0`` is
        allowed and returns the round PSF.
    angle_rad : float
        Trail position angle, ``atan2(vy, vx)`` — the source's direction of
        motion in image coordinates (row = y, col = x).
    truncate : float
        Gaussian half-extent in units of σ; the kernel is grown by L/2 to
        contain the trail at any orientation.
    """
    if sigma_px <= 0.0:
        raise ValueError(f"sigma_px must be > 0, got {sigma_px}")
    length = max(0.0, float(length_px))
    # Square support large enough for the trail (L/2) plus the Gaussian
    # skirt (truncate·σ) at any orientation.
    half = max(1, int(math.ceil(truncate * sigma_px + 0.5 * length)))
    ax = np.arange(-half, half + 1, dtype=np.float64)
    dx, dy = np.meshgrid(ax, ax)  # dx = column (x) offset, dy = row (y) offset
    ct, st = math.cos(angle_rad), math.sin(angle_rad)
    a = dx * ct + dy * st            # along-track
    c = -dx * st + dy * ct           # cross-track
    cross = np.exp(-(c**2) / (2.0 * sigma_px**2))
    if length <= 0.0:
        along = np.exp(-(a**2) / (2.0 * sigma_px**2))
    else:
        inv = 1.0 / (sigma_px * math.sqrt(2.0))
        along = 0.5 * (
            erf((a + 0.5 * length) * inv) - erf((a - 0.5 * length) * inv)
        )
    kernel = cross * along
    total = kernel.sum()
    if total <= 0.0:  # degenerate guard (never hit for σ > 0)
        return gaussian_psf_kernel(sigma_px, truncate)
    return kernel / total


def matched_streak_kernel(
    sigma_px: float,
    vx_px_s: float,
    vy_px_s: float,
    exposure_s: float,
    *,
    min_length_px: float = 0.0,
    truncate: float = 4.0,
) -> np.ndarray:
    """Velocity-matched kernel: round PSF, or a streak kernel if it trails.

    Convenience wrapper for the pipeline: given a velocity hypothesis and
    the sensor exposure, return the round :func:`gaussian_psf_kernel` when
    the within-exposure trail is negligible (``length < min_length_px``, or
    ``exposure_s <= 0``) and a :func:`streak_psf_kernel` matched to
    ``(vx, vy)`` otherwise.  Below the gate the result is *byte-identical*
    to the round kernel, so slow movers, short exposures and wide fields are
    unaffected.
    """
    length = streak_length_px(math.hypot(vx_px_s, vy_px_s), exposure_s)
    if length < max(0.0, min_length_px) or length <= 0.0:
        return gaussian_psf_kernel(sigma_px, truncate)
    return streak_psf_kernel(
        sigma_px, length, math.atan2(vy_px_s, vx_px_s), truncate
    )


def _config_exposure_s(cfg) -> float:
    """Effective exposure (s) for streak matching: 0 unless ``streak_kernel``."""
    if not bool(getattr(cfg, "streak_kernel", False)):
        return 0.0
    return float(getattr(cfg, "exposure_time_s", 0.0))


def temporal_median_subtract(
    frames: list[np.ndarray],
) -> tuple[list[np.ndarray], np.ndarray]:
    """Subtract the per-pixel temporal median (static scene) from each frame.

    For a fixed staring mount the temporal median over a stacking window is
    the static scene — stars, sky gradient, hot pixels — with no catalog
    dependency (assessment F5; Yanagisawa-style star elimination).  A LEO
    mover crosses ≫ PSF per window, occupying any given pixel in only a few
    frames, so it barely biases the median.

    That assumption FAILS for slow movers: below total motion ≈ 4 PSF-FWHM
    the source sits in the median and is subtracted with the static scene
    (:func:`median_subtraction_safe`).  Callers searching slow velocity
    bands must skip this step — :meth:`PsiPhiStacker.stack` and
    :func:`~opta_pipeline.coarse_fine.blind_coarse_fine_search` do so
    automatically unless overridden.

    Returns
    -------
    (subtracted_frames, median_image)
    """
    if not frames:
        raise ValueError("frames must not be empty")
    cube = np.stack([np.asarray(f, dtype=np.float64) for f in frames])
    median_image = np.median(cube, axis=0)
    return [f - median_image for f in cube], median_image


def make_psi_phi(
    frames: list[np.ndarray],
    noise_rms: float | Sequence[float],
    psf_kernel: np.ndarray,
    masks: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build per-frame ψ and φ likelihood images.

    Parameters
    ----------
    frames : list of np.ndarray
        Calibrated, background-subtracted frames (one shared shape).
    noise_rms : float | sequence of float
        Per-frame single-pixel noise σ (scalar applies to all frames).
        Frames with larger σ are automatically down-weighted ∝ 1/σ².
    psf_kernel : np.ndarray
        Unit-sum PSF kernel (:func:`gaussian_psf_kernel`).
    masks : list of np.ndarray | None
        Optional per-frame boolean arrays, True = invalid pixel (e.g.
        saturated).  Invalid pixels contribute neither signal nor weight
        (V = ∞).

    Returns
    -------
    (psi_list, phi_list)
    """
    n = len(frames)
    if n == 0:
        raise ValueError("frames must not be empty")
    if _is_scalar_noise_rms(noise_rms):
        sigmas = [float(noise_rms)] * n
    else:
        sigmas = [float(s) for s in noise_rms]
    if len(sigmas) != n:
        raise ValueError(f"len(noise_rms)={len(sigmas)} != len(frames)={n}")
    if any(s <= 0.0 for s in sigmas):
        raise ValueError("noise_rms entries must be > 0")

    kernel_sq = psf_kernel * psf_kernel
    psi_list: list[np.ndarray] = []
    phi_list: list[np.ndarray] = []
    for k, frame in enumerate(frames):
        data = np.asarray(frame, dtype=np.float64)
        valid = np.ones_like(data)
        if masks is not None and masks[k] is not None:
            valid = np.where(masks[k], 0.0, 1.0)
            data = data * valid
        inv_var = 1.0 / (sigmas[k] * sigmas[k])
        # mode="constant", cval=0 ⇒ out-of-frame pixels carry zero weight:
        # the φ image alone encodes per-pixel effective coverage.
        psi_list.append(
            ndimage_correlate(data * inv_var, psf_kernel, mode="constant", cval=0.0)
        )
        phi_list.append(
            ndimage_correlate(valid * inv_var, kernel_sq, mode="constant", cval=0.0)
        )
    return psi_list, phi_list


@dataclass(frozen=True)
class LikelihoodStackResult:
    """Result of a ψ/φ likelihood track-and-stack.

    Attributes
    ----------
    snr_map : np.ndarray
        Matched-filter SNR per reference-epoch pixel for the best velocity
        hypothesis.  Calibrated: ~ N(0, 1) under pure noise, so detection
        thresholds follow from trials statistics
        (:func:`poisson_tail_threshold`), not from empirical tuning.
    flux_map : np.ndarray
        ML point-source flux estimate Ψ/Φ (0 where Φ = 0).  Unbiased under
        partial coverage — usable for photometry.
    phi_map : np.ndarray
        Co-added Φ: the per-pixel statistical weight (effective coverage ×
        inverse variance).  Zero where no frame ever contributed.
    n_frames : int
        Number of frames stacked.
    best_vx_px_s, best_vy_px_s : float
        Winning velocity hypothesis (pixels/second).
    peak_snr : float
        ``snr_map.max()`` for the winning hypothesis.  Compare against a
        trials-corrected threshold, never the single-frame threshold
        (assessment F6).
    response_grid : np.ndarray | None
        Max-SNR per velocity hypothesis, shape (len(vx), len(vy)).
    median_image : np.ndarray | None
        Static-scene estimate that was subtracted, if enabled.
    """

    snr_map: np.ndarray = field(hash=False, compare=False)
    flux_map: np.ndarray = field(hash=False, compare=False)
    phi_map: np.ndarray = field(hash=False, compare=False)
    n_frames: int
    best_vx_px_s: float
    best_vy_px_s: float
    peak_snr: float
    response_grid: np.ndarray | None = field(
        default=None, hash=False, compare=False
    )
    median_image: np.ndarray | None = field(
        default=None, hash=False, compare=False
    )

    @property
    def peak_xy(self) -> tuple[int, int]:
        """(x, y) pixel of the SNR peak at the reference epoch."""
        r, c = np.unravel_index(int(np.argmax(self.snr_map)), self.snr_map.shape)
        return int(c), int(r)


class PsiPhiStacker:
    """Blind track-and-stack over a velocity grid using ψ/φ likelihoods.

    Drop-in conceptual replacement for :class:`~opta_pipeline.stack.Stacker`
    with a calibrated statistic.  The velocity-grid *sampling* requirements
    are identical (step ≈ PSF_fwhm / T_window — assessment F1); this class
    changes what is measured per hypothesis, not how many hypotheses are
    needed.

    Parameters
    ----------
    vx_grid, vy_grid : array-like
        Velocity hypotheses in pixels/second.
    psf_sigma_px : float
        Gaussian PSF σ for the matched-filter kernel.
    velocity_min_px_s : float
        Exclude hypotheses with |v| below this (static-source suppression);
        0 searches the full grid.  With temporal-median subtraction enabled
        static sources are already removed, so this is a belt-and-braces
        option rather than a necessity.
    """

    def __init__(
        self,
        vx_grid: np.ndarray,
        vy_grid: np.ndarray,
        psf_sigma_px: float = DEFAULT_PSF_SIGMA_PX,
        velocity_min_px_s: float = 0.0,
        *,
        exposure_s: float = 0.0,
        streak_min_length_px: float = 0.0,
    ) -> None:
        """Store the velocity grids and matched-filter kernel parameters.

        ``exposure_s`` > 0 enables the within-exposure streak kernel (F4):
        the velocity *search* still uses the round PSF, but the winning
        hypothesis is re-scored with a kernel matched to its trail
        ``|v|·exposure`` when that trail clears ``streak_min_length_px``.
        ``exposure_s = 0`` (default) disables it — pure round-PSF behaviour.
        """
        self.vx_grid = np.asarray(vx_grid, dtype=np.float64)
        self.vy_grid = np.asarray(vy_grid, dtype=np.float64)
        self.psf_sigma_px = float(psf_sigma_px)
        self.velocity_min_px_s = float(velocity_min_px_s)
        self.exposure_s = float(exposure_s)
        self.streak_min_length_px = float(streak_min_length_px)

    @classmethod
    def from_config(
        cls, config, psf_sigma_px: float | None = None
    ) -> PsiPhiStacker:
        """Build from a PipelineConfig / StackingConfig (same as Stacker).

        ``psf_sigma_px`` defaults to the config's own
        ``stacking.psf_sigma_px`` (the single production source of truth,
        pinned to the synth PSF regime σ = ``psf_fwhm_px`` / 2.355 ≈ 0.85 px
        in ``pipeline_defaults.yaml``); pass an explicit value only to
        override it.  It is an argument rather than a hard constant until a
        PSF model lands in the config.  The streak kernel is read from
        ``streak_kernel`` / ``exposure_time_s`` / ``streak_min_length_px``
        when present (opt-in; off by default).
        """
        cfg = getattr(config, "stacking", config)
        vmax = cfg.velocity_max_px_s
        vstep = cfg.velocity_step_px_s
        grid = symmetric_velocity_axis(vmax, vstep)
        vmin = float(getattr(cfg, "velocity_min_px_s", 0.0))
        if psf_sigma_px is None:
            psf_sigma_px = float(
                getattr(cfg, "psf_sigma_px", DEFAULT_PSF_SIGMA_PX)
            )
        return cls(
            vx_grid=grid,
            vy_grid=grid,
            psf_sigma_px=psf_sigma_px,
            velocity_min_px_s=vmin,
            exposure_s=_config_exposure_s(cfg),
            streak_min_length_px=float(getattr(cfg, "streak_min_length_px", 0.0)),
        )

    @classmethod
    def from_prior(
        cls,
        vx0_px_s: float,
        vy0_px_s: float,
        half_width_px_s: float,
        step_px_s: float,
        psf_sigma_px: float = DEFAULT_PSF_SIGMA_PX,
        velocity_min_px_s: float = 0.0,
        *,
        exposure_s: float = 0.0,
        streak_min_length_px: float = 0.0,
    ) -> PsiPhiStacker:
        """Grid bracketing a predicted velocity vector (same as Stacker)."""
        if step_px_s <= 0.0:
            raise ValueError(f"step_px_s must be > 0, got {step_px_s}")
        hw = max(0.0, float(half_width_px_s))
        offsets = symmetric_velocity_axis(hw, step_px_s)
        return cls(
            vx_grid=vx0_px_s + offsets,
            vy_grid=vy0_px_s + offsets,
            psf_sigma_px=psf_sigma_px,
            velocity_min_px_s=velocity_min_px_s,
            exposure_s=exposure_s,
            streak_min_length_px=streak_min_length_px,
        )

    def stack(
        self,
        frames: list[np.ndarray],
        noise_rms: float | Sequence[float],
        frame_times_s: list[float] | None = None,
        *,
        masks: list[np.ndarray] | None = None,
        subtract_temporal_median: bool | None = None,
    ) -> LikelihoodStackResult:
        """Search the velocity grid; return the best hypothesis' SNR map.

        Parameters
        ----------
        frames : list of np.ndarray
            Calibrated frames sharing one shape.
        noise_rms : float | sequence of float
            Per-frame single-pixel noise σ (scalar = same for all).
        frame_times_s : list of float | None
            Seconds relative to the reference epoch (t = 0 at midpass
            keeps trajectories in-frame).  None ⇒ t_k = k.
        masks : list of np.ndarray | None
            Optional per-frame True-=-invalid masks (see make_psi_phi).
        subtract_temporal_median : bool | None
            Remove the static scene (stars, gradients, hot pixels) before
            building ψ/φ.  None (default) decides automatically via
            :func:`median_subtraction_safe`: skipped when the whole
            velocity grid sits in the median's slow-mover blind spot.
            Pass False for pre-differenced input, True to force.
        """
        if not frames:
            raise ValueError("frames must not be empty")
        n = len(frames)
        if frame_times_s is None:
            frame_times_s = list(range(n))
        if len(frame_times_s) != n:
            raise ValueError(
                f"len(frame_times_s)={len(frame_times_s)} != len(frames)={n}"
            )
        times = np.asarray(frame_times_s, dtype=np.float64)

        if subtract_temporal_median is None:
            v_search_max = float(
                math.hypot(
                    np.abs(self.vx_grid).max() if self.vx_grid.size else 0.0,
                    np.abs(self.vy_grid).max() if self.vy_grid.size else 0.0,
                )
            )
            t_span = float(times.max() - times.min()) if n > 1 else 0.0
            subtract_temporal_median = median_subtraction_safe(
                v_search_max, t_span, self.psf_sigma_px
            )

        median_image: np.ndarray | None = None
        work_frames = [np.asarray(f, dtype=np.float64) for f in frames]
        if subtract_temporal_median:
            work_frames, median_image = temporal_median_subtract(work_frames)

        kernel = gaussian_psf_kernel(self.psf_sigma_px)
        psi_list, phi_list = make_psi_phi(work_frames, noise_rms, kernel, masks)

        hypotheses = [
            (i, j, float(vx), float(vy))
            for i, vx in enumerate(self.vx_grid)
            for j, vy in enumerate(self.vy_grid)
        ]
        vmin = self.velocity_min_px_s
        if vmin > 0.0:
            passed = [h for h in hypotheses if math.hypot(h[2], h[3]) >= vmin]
            if passed:
                hypotheses = passed

        shape = work_frames[0].shape
        response_grid = np.zeros(
            (len(self.vx_grid), len(self.vy_grid)), dtype=np.float64
        )
        best: tuple[float, float, float, np.ndarray, np.ndarray] | None = None
        for i, j, vx, vy in hypotheses:
            psi_sum = np.zeros(shape, dtype=np.float64)
            phi_sum = np.zeros(shape, dtype=np.float64)
            for k in range(n):
                dx = int(round(vx * times[k]))
                dy = int(round(vy * times[k]))
                _add_shifted(psi_sum, psi_list[k], dy, dx)
                _add_shifted(phi_sum, phi_list[k], dy, dx)
            snr_map = _snr_from(psi_sum, phi_sum)
            peak = float(snr_map.max()) if snr_map.size else 0.0
            response_grid[i, j] = peak
            if best is None or peak > best[0]:
                best = (peak, vx, vy, psi_sum, phi_sum)

        assert best is not None  # hypotheses is never empty
        peak_snr, best_vx, best_vy, psi_sum, phi_sum = best

        # Within-exposure streak re-score (F4): the search ranked hypotheses
        # with the round PSF (one kernel, cheap grid); now that the winning
        # velocity is fixed, re-measure it with a kernel matched to its trail
        # |v|·exposure.  Ψ/√Φ stays ~N(0,1) for this fixed kernel, so the
        # calibrated statistic is preserved.  Skipped (identical result) when
        # the trail is below streak_min_length_px.
        if self.exposure_s > 0.0:
            trail = streak_length_px(math.hypot(best_vx, best_vy), self.exposure_s)
            if trail >= self.streak_min_length_px and trail > 0.0:
                kernel_s = streak_psf_kernel(
                    self.psf_sigma_px, trail, math.atan2(best_vy, best_vx)
                )
                psi_s, phi_s = make_psi_phi(work_frames, noise_rms, kernel_s, masks)
                psi_sum = np.zeros(shape, dtype=np.float64)
                phi_sum = np.zeros(shape, dtype=np.float64)
                for k in range(n):
                    dx = int(round(best_vx * times[k]))
                    dy = int(round(best_vy * times[k]))
                    _add_shifted(psi_sum, psi_s[k], dy, dx)
                    _add_shifted(phi_sum, phi_s[k], dy, dx)
                peak_snr = float(_snr_from(psi_sum, phi_sum).max())

        snr_map = _snr_from(psi_sum, phi_sum)
        with np.errstate(divide="ignore", invalid="ignore"):
            flux_map = np.where(phi_sum > _PHI_EPS, psi_sum / phi_sum, 0.0)

        return LikelihoodStackResult(
            snr_map=snr_map,
            flux_map=flux_map,
            phi_map=phi_sum,
            n_frames=n,
            best_vx_px_s=best_vx,
            best_vy_px_s=best_vy,
            peak_snr=peak_snr,
            response_grid=response_grid,
            median_image=median_image,
        )


@dataclass(frozen=True)
class ForcedMeasurement:
    """Per-frame forced photometry at a predicted track position (F8).

    ``snr`` and ``flux_e`` are the matched-filter statistic and ML flux at
    the reported position.  ``refined`` is True when a local peak above the
    refinement threshold was found within the search radius — then
    ``x_px, y_px`` is the sub-pixel measured centroid (real astrometric
    information); otherwise the predicted position is passed through and
    the photometry is a forced upper-limit-style measurement.
    """

    x_px: float
    y_px: float
    snr: float
    flux_e: float
    refined: bool


def forced_measurement(
    frame: np.ndarray,
    noise_rms: float,
    psf_kernel: np.ndarray,
    x_pred: float,
    y_pred: float,
    *,
    search_radius_px: float = 2.0,
    refine_snr_min: float = 7.0,
) -> ForcedMeasurement:
    """Measure one frame at a predicted position (matched filter on a cutout).

    Computes ψ/φ on a small patch around ``(x_pred, y_pred)`` (frame must be
    background/static-scene subtracted, like all ψ/φ inputs).  If the local
    SNR peak within ``search_radius_px`` reaches ``refine_snr_min``, the
    returned position is the quadratic sub-pixel interpolation of that peak
    — a *measured* centroid with per-frame astrometric content.  Otherwise
    the prediction is returned unchanged with the forced statistic at the
    predicted pixel, so faint frames contribute honest low-SNR photometry
    instead of fabricated positions (assessment F8).

    ``refine_snr_min`` must sit well above the single-frame detection
    threshold: near ~3 sigma the maximum over the search disc is dominated
    by noise, so refinement would snap a marginal target onto noise peaks
    and corrupt the track astrometry (e2e findings #11/#12).  Default 7.0
    matches ``StackingConfig.refine_snr_min`` (see its comment for the
    margin over the nominal ~5 sigma).
    """
    h, w = frame.shape
    kh = psf_kernel.shape[0] // 2
    r = int(math.ceil(search_radius_px))
    half = kh + r + 1
    cx = int(round(x_pred))
    cy = int(round(y_pred))

    y0, y1 = cy - half, cy + half + 1
    x0, x1 = cx - half, cx + half + 1
    py0, py1 = max(0, y0), min(h, y1)
    px0, px1 = max(0, x0), min(w, x1)
    if py1 <= py0 or px1 <= px0:
        return ForcedMeasurement(x_pred, y_pred, 0.0, 0.0, False)

    size = 2 * half + 1
    patch = np.zeros((size, size))
    valid = np.zeros((size, size))
    patch[py0 - y0 : py1 - y0, px0 - x0 : px1 - x0] = frame[py0:py1, px0:px1]
    valid[py0 - y0 : py1 - y0, px0 - x0 : px1 - x0] = 1.0

    inv_var = 1.0 / max(noise_rms, 1e-12) ** 2
    psi = ndimage_correlate(patch * inv_var, psf_kernel, mode="constant", cval=0.0)
    phi = ndimage_correlate(
        valid * inv_var, psf_kernel * psf_kernel, mode="constant", cval=0.0
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(phi > _PHI_EPS, psi / np.sqrt(phi), 0.0)

    # Search disc around the patch centre (the predicted pixel).
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    in_disc = (yy - half) ** 2 + (xx - half) ** 2 <= search_radius_px**2
    disc_snr = np.where(in_disc, snr, -np.inf)
    iy, ix = np.unravel_index(int(np.argmax(disc_snr)), disc_snr.shape)
    peak_val = float(disc_snr[iy, ix])

    if peak_val >= refine_snr_min and 0 < iy < size - 1 and 0 < ix < size - 1:
        # Quadratic sub-pixel interpolation around the local SNR peak.
        dy = _parabolic_offset(snr[iy - 1, ix], snr[iy, ix], snr[iy + 1, ix])
        dx = _parabolic_offset(snr[iy, ix - 1], snr[iy, ix], snr[iy, ix + 1])
        flux = float(psi[iy, ix] / phi[iy, ix]) if phi[iy, ix] > _PHI_EPS else 0.0
        return ForcedMeasurement(
            x_px=cx + (ix - half) + dx,
            y_px=cy + (iy - half) + dy,
            snr=peak_val,
            flux_e=flux,
            refined=True,
        )

    # Forced: statistic at the predicted pixel, position passed through.
    fy = min(max(int(round(y_pred)) - y0, 0), size - 1)
    fx = min(max(int(round(x_pred)) - x0, 0), size - 1)
    flux = float(psi[fy, fx] / phi[fy, fx]) if phi[fy, fx] > _PHI_EPS else 0.0
    return ForcedMeasurement(
        x_px=x_pred,
        y_px=y_pred,
        snr=float(snr[fy, fx]),
        flux_e=flux,
        refined=False,
    )


def _parabolic_offset(left: float, centre: float, right: float) -> float:
    """Sub-pixel offset of a parabola through three samples, clipped to ±0.5."""
    denom = left - 2.0 * centre + right
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))


def _snr_from(psi_sum: np.ndarray, phi_sum: np.ndarray) -> np.ndarray:
    """SNR map Ψ/√Φ, zero where the statistical weight Φ vanishes."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            phi_sum > _PHI_EPS, psi_sum / np.sqrt(phi_sum), 0.0
        )


def _add_shifted(
    accumulator: np.ndarray, image: np.ndarray, dy: int, dx: int
) -> None:
    """Accumulate ``image`` shifted by (−dy, −dx) — integer, no wraparound.

    Registers a source at (y₀ + dy, x₀ + dx) onto (y₀, x₀):
    ``accumulator[y, x] += image[y + dy, x + dx]`` over the valid overlap.
    Pure slice arithmetic: ~100× cheaper than interpolating shifts and
    exact for noise statistics (no correlation introduced).
    """
    h, w = accumulator.shape
    src_y0, src_y1 = max(0, dy), min(h, h + dy)
    dst_y0, dst_y1 = max(0, -dy), min(h, h - dy)
    src_x0, src_x1 = max(0, dx), min(w, w + dx)
    dst_x0, dst_x1 = max(0, -dx), min(w, w - dx)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return
    accumulator[dst_y0:dst_y1, dst_x0:dst_x1] += image[
        src_y0:src_y1, src_x0:src_x1
    ]
