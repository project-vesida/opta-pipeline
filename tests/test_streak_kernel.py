"""Tests for the within-exposure streak (motion-blur) matched-filter kernel (F4).

A source moving at |v| px/s trails |v|·exposure px within one frame.  Matched-
filtering that trail with the round point PSF loses SNR; a kernel matched to the
trail recovers it while keeping the ψ/φ statistic calibrated (Ψ/√Φ ~ N(0,1)).
The feature is opt-in: below ``streak_min_length_px`` — and whenever the exposure
is 0 — every code path is byte-identical to the round-PSF result.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from opta_pipeline.coarse_fine import blind_coarse_fine_search
from opta_pipeline.config import PipelineConfig
from opta_pipeline.likelihood import (
    PsiPhiStacker,
    _add_shifted,
    _snr_from,
    gaussian_psf_kernel,
    make_psi_phi,
    matched_streak_kernel,
    streak_length_px,
    streak_psf_kernel,
)

_SIGMA = 1.5


def _trailed_frames(
    roi, times, x0, y0, vx, vy, amp, *, sigma, exposure, noise, seed
):
    """Frames of a source on a linear track, trailed |v|·exposure within each.

    Each frame draws the source at its trajectory position (x0+vx·t, y0+vy·t)
    as a uniform line of length |v|·exposure along the motion direction,
    convolved with the σ PSF (approximated by dense sampling).
    """
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:roi, 0:roi].astype(np.float64)
    trail = math.hypot(vx, vy) * exposure
    ang = math.atan2(vy, vx)
    ux, uy = math.cos(ang), math.sin(ang)
    n_samp = max(1, int(math.ceil(trail)) * 3)
    offs = np.linspace(-trail / 2.0, trail / 2.0, n_samp)
    frames = []
    for t in times:
        f = rng.normal(0.0, noise, (roi, roi)) if noise > 0 else np.zeros((roi, roi))
        cx, cy = x0 + vx * t, y0 + vy * t
        blob = np.zeros((roi, roi))
        for s in offs:
            px, py = cx + s * ux, cy + s * uy
            blob += np.exp(-(((xs - px) ** 2 + (ys - py) ** 2) / (2.0 * sigma**2)))
        mx = float(blob.max())
        if mx > 0.0:
            blob *= amp / mx  # peak-normalize; skip when track is off-frame
        frames.append(f + blob)
    return frames


# --------------------------------------------------------------------------- #
# Kernel construction                                                         #
# --------------------------------------------------------------------------- #
class TestKernelConstruction:
    def test_reduces_to_gaussian_at_zero_length(self) -> None:
        """L → 0 recovers the round PSF exactly (strict generalization)."""
        for ang in (0.0, 0.6, 1.9, -2.3):
            assert np.allclose(
                streak_psf_kernel(_SIGMA, 0.0, ang),
                gaussian_psf_kernel(_SIGMA),
                atol=1e-12,
            )

    @pytest.mark.parametrize("length", [2.0, 8.0, 16.0])
    @pytest.mark.parametrize("ang_deg", [0.0, 30.0, 90.0, 145.0])
    def test_unit_sum(self, length, ang_deg) -> None:
        k = streak_psf_kernel(_SIGMA, length, math.radians(ang_deg))
        assert k.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.all(k >= 0.0)

    def test_elongated_along_motion(self) -> None:
        """Second moment is larger along the trail than across it."""
        ang = math.radians(30.0)
        k = streak_psf_kernel(_SIGMA, 10.0, ang)
        half = k.shape[0] // 2
        ax = np.arange(-half, half + 1, dtype=float)
        dx, dy = np.meshgrid(ax, ax)
        ct, st = math.cos(ang), math.sin(ang)
        along = (k * (dx * ct + dy * st) ** 2).sum()
        cross = (k * (-dx * st + dy * ct) ** 2).sum()
        assert along > 3.0 * cross

    def test_streak_length_and_optics_scaling(self) -> None:
        """Trail is |v|·exposure; a narrow field (more px/s) trails more."""
        assert streak_length_px(200.0, 0.04) == pytest.approx(8.0)
        assert streak_length_px(0.0, 0.04) == 0.0
        assert streak_length_px(200.0, 0.0) == 0.0
        # Same angular rate, narrower plate scale ⇒ higher px/s ⇒ longer trail.
        wide_px_s, narrow_px_s = 90.0, 260.0  # e.g. 0.55 deg/s at 22 vs 7.6 "/px
        assert streak_length_px(narrow_px_s, 0.04) > streak_length_px(
            wide_px_s, 0.04
        )

    def test_matched_kernel_gates_on_min_length(self) -> None:
        # Trail 0.4 px < 1.0 gate ⇒ round PSF; trail 8 px ⇒ streak kernel.
        below = matched_streak_kernel(_SIGMA, 10.0, 0.0, 0.04, min_length_px=1.0)
        above = matched_streak_kernel(_SIGMA, 200.0, 0.0, 0.04, min_length_px=1.0)
        assert np.allclose(below, gaussian_psf_kernel(_SIGMA))
        assert above.shape[0] > gaussian_psf_kernel(_SIGMA).shape[0]

    def test_matched_kernel_round_when_no_exposure(self) -> None:
        assert np.allclose(
            matched_streak_kernel(_SIGMA, 400.0, 0.0, 0.0), gaussian_psf_kernel(_SIGMA)
        )


# --------------------------------------------------------------------------- #
# Matched-filter behaviour                                                     #
# --------------------------------------------------------------------------- #
class TestMatchedFilter:
    def test_recovers_snr_on_trailed_source(self) -> None:
        """Streak kernel beats the round PSF on a real trailed source."""
        roi, n, noise = 64, 12, 1.0
        times = [0.0] * n  # co-registered; isolates the kernel effect
        trail = 10.0
        base = _trailed_frames(
            roi, times, 32, 32, trail, 0.0, 4.0,
            sigma=_SIGMA, exposure=1.0, noise=0.0, seed=0,
        )
        ang = 0.0
        kr, ks = gaussian_psf_kernel(_SIGMA), streak_psf_kernel(_SIGMA, trail, ang)

        def peak(kernel, seed):
            r = np.random.default_rng(seed)
            fr = [b + r.normal(0, noise, (roi, roi)) for b in base]
            psi, phi = make_psi_phi(fr, noise, kernel)
            ps, ph = np.zeros((roi, roi)), np.zeros((roi, roi))
            for k in range(n):
                _add_shifted(ps, psi[k], 0, 0)
                _add_shifted(ph, phi[k], 0, 0)
            return float(_snr_from(ps, ph).max())

        rr = np.mean([peak(kr, s) for s in range(24)])
        ss = np.mean([peak(ks, s) for s in range(24)])
        assert ss > 1.2 * rr  # ≳20% recovery on a 10 px trail

    def test_preserves_noise_calibration(self) -> None:
        """Ψ/√Φ under pure noise stays ~N(0,1) for the streak kernel."""
        roi, n = 96, 16
        ks = streak_psf_kernel(_SIGMA, 12.0, math.radians(40))
        stds = []
        for seed in range(12):
            r = np.random.default_rng(500 + seed)
            fr = [r.normal(0, 1.0, (roi, roi)) for _ in range(n)]
            psi, phi = make_psi_phi(fr, 1.0, ks)
            m = _snr_from(sum(psi), sum(phi))[12:-12, 12:-12]
            stds.append(float(m.std()))
        assert abs(np.mean(stds) - 1.0) < 0.05


# --------------------------------------------------------------------------- #
# PsiPhiStacker winner re-score                                               #
# --------------------------------------------------------------------------- #
class TestStackerRescore:
    def _scene(self, exposure, noise=0.0, seed=1):
        roi, n = 80, 14
        times = [(k - (n - 1) / 2.0) * 0.1 for k in range(n)]
        vx, vy = 120.0, 40.0  # trail = |v|·exposure
        frames = _trailed_frames(
            roi, times, 40, 40, vx, vy, 5.0,
            sigma=_SIGMA, exposure=exposure, noise=noise, seed=seed,
        )
        return frames, times, vx, vy

    def test_rescore_boosts_trailed_mover(self) -> None:
        frames, times, vx, vy = self._scene(exposure=0.06)
        common = dict(psf_sigma_px=_SIGMA)
        base = PsiPhiStacker.from_prior(vx, vy, 0.0, 20.0, **common).stack(
            frames, 1.0, times, subtract_temporal_median=False
        )
        streak = PsiPhiStacker.from_prior(
            vx, vy, 0.0, 20.0, exposure_s=0.06, streak_min_length_px=1.0, **common
        ).stack(frames, 1.0, times, subtract_temporal_median=False)
        assert streak.peak_snr > 1.1 * base.peak_snr

    def test_rescore_is_noop_below_gate(self) -> None:
        """Tiny exposure ⇒ trail below gate ⇒ identical result to disabled."""
        frames, times, vx, vy = self._scene(exposure=0.001)
        off = PsiPhiStacker.from_prior(vx, vy, 0.0, 20.0, psf_sigma_px=_SIGMA).stack(
            frames, 1.0, times, subtract_temporal_median=False
        )
        on = PsiPhiStacker.from_prior(
            vx, vy, 0.0, 20.0, psf_sigma_px=_SIGMA,
            exposure_s=0.001, streak_min_length_px=1.0,
        ).stack(frames, 1.0, times, subtract_temporal_median=False)
        assert on.peak_snr == pytest.approx(off.peak_snr, rel=1e-12)
        assert np.allclose(on.snr_map, off.snr_map)


# --------------------------------------------------------------------------- #
# Coarse-to-fine blind search gate                                            #
# --------------------------------------------------------------------------- #
class TestCoarseFineGate:
    def _scene(self, exposure, seed=3):
        # Track stays in-ROI over the window: x ∈ [64−42, 64+42] at ±0.7 s.
        roi, n = 128, 15
        times = [(k - (n - 1) / 2.0) * 0.1 for k in range(n)]
        vx, vy = 60.0, 0.0
        frames = _trailed_frames(
            roi, times, 64, 64, vx, vy, 5.0,
            sigma=_SIGMA, exposure=exposure, noise=1.0, seed=seed,
        )
        return frames, times

    def _search(self, frames, times, **kw):
        return blind_coarse_fine_search(
            frames, 1.0, times,
            v_max_px_s=120.0, coarse_step_px_s=20.0, psf_sigma_px=_SIGMA,
            subtract_median=False, **kw,
        )

    def test_streak_rescore_boosts_candidate_snr(self) -> None:
        frames, times = self._scene(exposure=0.12)
        base = self._search(frames, times)
        streak = self._search(
            frames, times, exposure_time_s=0.12, streak_min_length_px=1.0
        )
        assert base.candidates and streak.candidates
        assert streak.candidates[0].snr > base.candidates[0].snr

    def test_streak_noop_below_gate(self) -> None:
        """Trail below the gate ⇒ candidate SNRs unchanged vs disabled."""
        frames, times = self._scene(exposure=0.002)
        base = self._search(frames, times)
        streak = self._search(
            frames, times, exposure_time_s=0.002, streak_min_length_px=1.0
        )
        assert len(base.candidates) == len(streak.candidates)
        for a, b in zip(base.candidates, streak.candidates):
            assert a.snr == pytest.approx(b.snr, rel=1e-12)


# --------------------------------------------------------------------------- #
# Config wiring                                                                #
# --------------------------------------------------------------------------- #
class TestConfig:
    def test_streak_defaults_off(self) -> None:
        stk = PipelineConfig.default().stacking
        assert stk.streak_kernel is False
        # exposure_time_s is now a FALLBACK ONLY (default 0.0): the pipeline
        # derives the streak-kernel exposure from the actual frame cadence
        # (median inter-frame dt = frame period), one source of truth that
        # tracks the real capture mode (21 fps full-res / 25 fps ROI) instead
        # of the old hardcoded 1/25 s constant, which underestimated the trail
        # ~17% for the 21 fps detection basis.  See pipeline._streak_exposure_s.
        assert stk.exposure_time_s == pytest.approx(0.0)
        assert stk.streak_min_length_px == pytest.approx(1.0)
