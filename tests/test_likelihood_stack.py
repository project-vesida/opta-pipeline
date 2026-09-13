"""Tests for ψ/φ likelihood track-and-stack (Phase 1 of the stack rework).

Every quantitative assertion is checked against an analytic prediction of
the matched-filter statistic, not against pinned empirical values:

    SNR_pred = F_tot · √(ΣP²) · √N_vis / σ

for a source of total flux F_tot (= A·2πσ_psf² for an amplitude-A
Gaussian), unit-sum kernel P, and N_vis frames of visibility.

Structure
---------
TestKernelAndInputs      — kernel normalization, input validation
TestCalibration          — pure noise ⇒ snr_map ~ N(0, 1)  (the F3/F6 fix)
TestPointSourceRecovery  — SNR/flux/velocity match prediction; beats Stacker
TestInverseVariance      — bad frames are down-weighted, not poisonous
TestCoverage             — partial visibility: correct N_vis, unbiased flux
TestTemporalMedian       — static scene removed, mover preserved (F5)
TestSpeed                — integer-shift search is faster than interpolation
TestPoissonTailThreshold — Bernstein u* contract (2026-07-28): Gaussian
                           limit, Bernstein-equation inversion, monotonicity
TestBoundScaleEpsilon    — ε = b/√Φ estimator: closed form, per-frame max
TestQuantisationStep     — recorded-ladder estimator: floor, ladder, λ = 25
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from opta_pipeline.likelihood import (
    PsiPhiStacker,
    bound_scale_epsilon,
    gaussian_psf_kernel,
    make_psi_phi,
    median_subtraction_safe,
    poisson_tail_threshold,
    quantisation_step_e,
    temporal_median_subtract,
)
from opta_pipeline.stack import Stacker

_H, _W = 96, 96
_N = 25
_PSF_SIGMA = 1.2


def _times() -> list[float]:
    """25 midpass-centred times spanning ±2.4 s (0.2 s cadence)."""
    return [(k - (_N - 1) / 2.0) * 0.2 for k in range(_N)]


def _flux_total(amplitude: float, sigma: float = _PSF_SIGMA) -> float:
    """Total flux of an amplitude-A Gaussian source."""
    return amplitude * 2.0 * np.pi * sigma * sigma


def _snr_pred(
    amplitude: float, n_vis: int, noise: float, sigma: float = _PSF_SIGMA
) -> float:
    """Analytic matched-filter SNR prediction (module docstring formula)."""
    kernel = gaussian_psf_kernel(sigma)
    return (
        _flux_total(amplitude, sigma)
        * float(np.sqrt((kernel * kernel).sum()))
        * np.sqrt(n_vis)
        / noise
    )


def _moving_source_frames(
    amplitude: float,
    vx: float,
    vy: float,
    noise: float,
    seed: int,
    x0: float = _W / 2.0,
    y0: float = _H / 2.0,
) -> list[np.ndarray]:
    """Gaussian source on trajectory (x0 + vx·t, y0 + vy·t) plus noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:_H, 0:_W].astype(np.float64)
    frames = []
    for t in _times():
        xc, yc = x0 + vx * t, y0 + vy * t
        r2 = (xx - xc) ** 2 + (yy - yc) ** 2
        source = amplitude * np.exp(-r2 / (2.0 * _PSF_SIGMA**2))
        frames.append(source + rng.normal(0.0, noise, (_H, _W)))
    return frames


class TestKernelAndInputs:
    """Kernel normalization and input validation."""

    def test_kernel_is_unit_sum(self) -> None:
        kernel = gaussian_psf_kernel(1.7)
        assert kernel.sum() == pytest.approx(1.0, rel=1e-9)

    def test_kernel_rejects_nonpositive_sigma(self) -> None:
        with pytest.raises(ValueError):
            gaussian_psf_kernel(0.0)

    def test_empty_frames_raise(self) -> None:
        with pytest.raises(ValueError):
            PsiPhiStacker(np.array([0.0]), np.array([0.0])).stack([], 1.0)

    def test_noise_length_mismatch_raises(self) -> None:
        frames = [np.zeros((8, 8)) for _ in range(3)]
        with pytest.raises(ValueError):
            make_psi_phi(frames, [1.0, 1.0], gaussian_psf_kernel(1.0))

    def test_nonpositive_noise_raises(self) -> None:
        frames = [np.zeros((8, 8))]
        with pytest.raises(ValueError):
            make_psi_phi(frames, 0.0, gaussian_psf_kernel(1.0))

    @pytest.mark.parametrize(
        "scalar",
        [
            2.0,
            np.float64(2.0),
            np.float32(2.0),
            np.int64(2),
            np.int32(2),
            np.asarray(2.0),  # 0-d array: np.mean(...) / arr[()] shape
            np.asarray(2, dtype=np.int64),
        ],
        ids=[
            "float", "float64", "float32", "int64", "int32",
            "0d-float-array", "0d-int-array",
        ],
    )
    def test_numpy_scalar_noise_rms_is_a_scalar(self, scalar) -> None:
        """Every NumPy scalar σ must take the scalar branch, not the sequence one.

        ``float(arr.std())`` gives np.float64, a float32 calibration path gives
        np.float32, and an integer noise floor gives np.int64 — a predicate
        that misses any of them falls through to ``noise_rms[i]`` on a 0-d
        value and dies with IndexError/TypeError instead of stacking.
        """
        frames = [np.zeros((8, 8)) for _ in range(3)]
        psi, phi = make_psi_phi(frames, scalar, gaussian_psf_kernel(1.0))
        ref_psi, ref_phi = make_psi_phi(frames, 2.0, gaussian_psf_kernel(1.0))
        assert len(psi) == 3
        for got, want in zip(phi, ref_phi):
            np.testing.assert_allclose(got, want)
        for got, want in zip(psi, ref_psi):
            np.testing.assert_allclose(got, want)


class TestCalibration:
    """Pure noise ⇒ snr_map ~ N(0, 1): thresholds become pure statistics.

    This is the property the classical Stacker lacks (F3/F6): its peak
    statistic needs empirical calibration per geometry, while the ψ/φ SNR
    is directly comparable against √(2 ln N_trials).
    """

    @pytest.mark.parametrize("noise", [1.0, 3.7])
    def test_snr_map_is_standard_normal(self, noise: float) -> None:
        rng = np.random.default_rng(11)
        frames = [rng.normal(0.0, noise, (_H, _W)) for _ in range(_N)]
        stacker = PsiPhiStacker(
            np.array([30.0]), np.array([0.0]), psf_sigma_px=_PSF_SIGMA
        )
        result = stacker.stack(
            frames, noise, _times(), subtract_temporal_median=False
        )
        # Matched filtering correlates neighbouring pixels over ~4πσ_psf²,
        # so the 76×76 window holds only ~320 independent samples: the
        # sample mean has SE ≈ 0.056 (bound at 3·SE) while the sample std
        # is far more stable.  The std is the calibration claim.
        inner = result.snr_map[10:-10, 10:-10]
        assert abs(float(inner.mean())) < 0.17
        assert abs(float(inner.std()) - 1.0) < 0.05

    def test_variable_noise_still_calibrated(self) -> None:
        """Mixed per-frame σ must not break the N(0,1) property."""
        rng = np.random.default_rng(12)
        sigmas = [1.0 if k % 2 == 0 else 5.0 for k in range(_N)]
        frames = [rng.normal(0.0, s, (_H, _W)) for s in sigmas]
        stacker = PsiPhiStacker(
            np.array([30.0]), np.array([0.0]), psf_sigma_px=_PSF_SIGMA
        )
        result = stacker.stack(
            frames, sigmas, _times(), subtract_temporal_median=False
        )
        inner = result.snr_map[10:-10, 10:-10]
        assert abs(float(inner.mean())) < 0.17  # 3·SE, see above
        assert abs(float(inner.std()) - 1.0) < 0.05


class TestPointSourceRecovery:
    """Recovery of an in-frame moving source matches the analytic SNR."""

    _A = 8.0
    _VX = 10.0  # ±24 px about centre over ±2.4 s — always in frame

    def _stack(self):
        frames = _moving_source_frames(self._A, self._VX, 0.0, 1.0, seed=21)
        stacker = PsiPhiStacker(
            np.arange(0.0, 21.0, 5.0),
            np.array([0.0]),
            psf_sigma_px=_PSF_SIGMA,
        )
        return frames, stacker.stack(
            frames, 1.0, _times(), subtract_temporal_median=False
        )

    def test_velocity_and_position_recovered(self) -> None:
        _, result = self._stack()
        assert result.best_vx_px_s == pytest.approx(self._VX)
        x, y = result.peak_xy
        assert abs(x - _W / 2) <= 1 and abs(y - _H / 2) <= 1

    def test_snr_matches_matched_filter_prediction(self) -> None:
        _, result = self._stack()
        pred = _snr_pred(self._A, _N, 1.0)
        assert result.peak_snr == pytest.approx(pred, rel=0.12)

    def test_flux_estimate_is_unbiased(self) -> None:
        _, result = self._stack()
        x, y = result.peak_xy
        assert result.flux_map[y, x] == pytest.approx(
            _flux_total(self._A), rel=0.15
        )

    def test_beats_classical_peak_pixel_snr(self) -> None:
        """Matched filter gain over peak-pixel scoring (~2× at σ_psf=1.2).

        Same frames, same grid, same times: the ψ/φ SNR must exceed the
        classical Stacker peak statistic — the F4(point) gain.
        """
        frames, result = self._stack()
        classical = Stacker(
            np.arange(0.0, 21.0, 5.0), np.array([0.0])
        ).stack(frames, 1.0, _times())
        assert classical.best_vx_px_s == pytest.approx(self._VX)
        assert result.peak_snr > 1.5 * classical.peak_snr


class TestInverseVariance:
    """Bad frames are down-weighted ∝1/σ², never poisonous."""

    def test_noisy_frames_do_not_poison_the_stack(self) -> None:
        amplitude, vx = 8.0, 10.0
        rng = np.random.default_rng(31)
        yy, xx = np.mgrid[0:_H, 0:_W].astype(np.float64)
        sigmas = [1.0] * 12 + [10.0] * 13
        frames = []
        for t, s in zip(_times(), sigmas):
            xc = _W / 2.0 + vx * t
            r2 = (xx - xc) ** 2 + (yy - _H / 2.0) ** 2
            frames.append(
                amplitude * np.exp(-r2 / (2.0 * _PSF_SIGMA**2))
                + rng.normal(0.0, s, (_H, _W))
            )
        stacker = PsiPhiStacker(
            np.arange(0.0, 21.0, 5.0), np.array([0.0]), psf_sigma_px=_PSF_SIGMA
        )
        result = stacker.stack(
            frames, sigmas, _times(), subtract_temporal_median=False
        )
        # Effective frame count Σ(σ_ref/σ_k)² = 12 + 13/100 ≈ 12.13: the
        # 13 bad frames contribute ~1% weight instead of drowning the
        # stack (which is what unweighted summation would do).
        kernel = gaussian_psf_kernel(_PSF_SIGMA)
        pred = (
            _flux_total(amplitude)
            * float(np.sqrt((kernel * kernel).sum()))
            * float(np.sqrt(sum(1.0 / (s * s) for s in sigmas)))
        )
        assert result.best_vx_px_s == pytest.approx(vx)
        assert result.peak_snr == pytest.approx(pred, rel=0.2)
        # ... and is at least ~90% of what the good frames alone give.
        good_only = _snr_pred(amplitude, 12, 1.0)
        assert result.peak_snr > 0.85 * good_only


class TestCoverage:
    """Partial visibility: SNR reflects N_vis, flux stays unbiased (F3)."""

    _A = 8.0
    _VX = 30.0  # source in frame for only ~16 of 25 frames

    def _result(self):
        frames = _moving_source_frames(self._A, self._VX, 0.0, 1.0, seed=41)
        stacker = PsiPhiStacker(
            np.arange(20.0, 41.0, 5.0), np.array([0.0]), psf_sigma_px=_PSF_SIGMA
        )
        return stacker.stack(
            frames, 1.0, _times(), subtract_temporal_median=False
        )

    def test_snr_scales_with_visible_frames_only(self) -> None:
        result = self._result()
        # In-frame condition 0 ≤ 48 + 30·t < 96 ⇒ 16 visible frames.
        n_vis = sum(1 for t in _times() if 0 <= _W / 2 + self._VX * t < _W)
        assert n_vis == 16
        pred = _snr_pred(self._A, n_vis, 1.0)
        assert result.peak_snr == pytest.approx(pred, rel=0.15)

    def test_flux_unbiased_under_partial_coverage(self) -> None:
        """Ψ/Φ normalizes by actual weight — the classical sum cannot."""
        result = self._result()
        x, y = result.peak_xy
        assert result.flux_map[y, x] == pytest.approx(
            _flux_total(self._A), rel=0.2
        )

    def test_phi_zero_where_never_covered(self) -> None:
        """φ encodes per-pixel coverage; uncovered pixels have zero weight
        and therefore SNR exactly 0 — no edge false alarms (F3)."""
        result = self._result()
        assert np.all(result.snr_map[result.phi_map == 0.0] == 0.0)


class TestTemporalMedian:
    """Static-scene subtraction (F5): stars out, mover preserved."""

    _A_STAR = 50.0
    _A_MOVER = 8.0
    _VX = 10.0
    _STAR_XY = (30, 60)  # (x, y)

    def _frames(self) -> list[np.ndarray]:
        rng = np.random.default_rng(51)
        yy, xx = np.mgrid[0:_H, 0:_W].astype(np.float64)
        sx, sy = self._STAR_XY
        star = self._A_STAR * np.exp(
            -((xx - sx) ** 2 + (yy - sy) ** 2) / (2.0 * _PSF_SIGMA**2)
        )
        frames = []
        for t in _times():
            xc = _W / 2.0 + self._VX * t
            r2 = (xx - xc) ** 2 + (yy - _H / 2.0) ** 2
            mover = self._A_MOVER * np.exp(-r2 / (2.0 * _PSF_SIGMA**2))
            frames.append(star + mover + rng.normal(0.0, 1.0, (_H, _W)))
        return frames

    def _grid(self) -> PsiPhiStacker:
        return PsiPhiStacker(
            np.arange(0.0, 21.0, 5.0), np.array([0.0]), psf_sigma_px=_PSF_SIGMA
        )

    def test_star_wins_without_subtraction(self) -> None:
        """Motivation: with the static scene left in, the v=0 hypothesis
        containing the bright star dominates the search."""
        result = self._grid().stack(
            self._frames(), 1.0, _times(), subtract_temporal_median=False
        )
        x, y = result.peak_xy
        assert result.best_vx_px_s == pytest.approx(0.0)
        assert (x, y) == self._STAR_XY

    def test_mover_wins_with_subtraction(self) -> None:
        result = self._grid().stack(
            self._frames(), 1.0, _times(), subtract_temporal_median=True
        )
        assert result.best_vx_px_s == pytest.approx(self._VX)
        x, y = result.peak_xy
        assert abs(x - _W / 2) <= 1 and abs(y - _H / 2) <= 1
        # Mover SNR survives subtraction nearly intact (occupies any given
        # pixel in ≤2 of 25 frames ⇒ median untouched)...
        assert result.peak_snr > 0.85 * _snr_pred(self._A_MOVER, _N, 1.0)
        # ...while the star position is quiet in the SNR map.
        sx, sy = self._STAR_XY
        assert abs(result.snr_map[sy, sx]) < 5.0

    def test_median_image_is_the_static_scene(self) -> None:
        _, median_image = temporal_median_subtract(self._frames())
        sx, sy = self._STAR_XY
        assert median_image[sy, sx] == pytest.approx(self._A_STAR, rel=0.1)


class TestMedianBlindSpotGuard:
    """Slow movers sit in the temporal median and get subtracted with it.

    Regression for the 2026-07-02 smoke regression: a target with total
    motion v·T ≲ 4 PSF-FWHM is quasi-static for the median, so blanket
    subtraction destroyed the signal (validation smoke scene: 5 px/s ×
    0.44 s = 2.2 px against a 2.8 px FWHM → detected=False).  ``stack``
    must skip the subtraction automatically when the *whole search band*
    is inside that blind spot.
    """

    _V_SLOW = 5.0
    _SLOW_N = 12

    def _slow_times(self) -> list[float]:
        """12 midpass-centred times spanning ±0.22 s (25 fps)."""
        return [(k - (self._SLOW_N - 1) / 2.0) * 0.04 for k in range(self._SLOW_N)]

    def _slow_frames(self, amplitude: float = 12.0) -> list[np.ndarray]:
        rng = np.random.default_rng(7)
        yy, xx = np.mgrid[0:_H, 0:_W].astype(np.float64)
        frames = []
        for t in self._slow_times():
            xc = _W / 2.0 + self._V_SLOW * t
            r2 = (xx - xc) ** 2 + (yy - _H / 2.0) ** 2
            frames.append(
                amplitude * np.exp(-r2 / (2.0 * _PSF_SIGMA**2))
                + rng.normal(0.0, 1.0, (_H, _W))
            )
        return frames

    def _slow_grid(self) -> PsiPhiStacker:
        return PsiPhiStacker(
            np.arange(-10.0, 11.0, 5.0),
            np.arange(-10.0, 11.0, 5.0),
            psf_sigma_px=_PSF_SIGMA,
        )

    def test_safe_predicate(self) -> None:
        fwhm = 2.355 * 1.5
        # Validation smoke geometry: ±10 px/s over 0.44 s — blind.
        assert not median_subtraction_safe(10.0, 0.44, 1.5)
        # Operational geometry: 60 px/s over 2 s — safe.
        assert median_subtraction_safe(60.0, 2.0, 1.5)
        # Boundary is inclusive at exactly 4 FWHM of motion.
        assert median_subtraction_safe(4.0 * fwhm, 1.0, 1.5)

    def test_auto_skips_subtraction_for_slow_band(self) -> None:
        result = self._slow_grid().stack(self._slow_frames(), 1.0, self._slow_times())
        assert result.median_image is None  # guard fired: no subtraction
        assert result.best_vx_px_s == pytest.approx(self._V_SLOW)
        assert result.best_vy_px_s == pytest.approx(0.0)
        assert result.peak_snr > 0.8 * _snr_pred(12.0, self._SLOW_N, 1.0)

    def test_forced_subtraction_reproduces_the_blind_spot(self) -> None:
        """Documents the failure the guard prevents (do not 'fix' by
        retuning the scene — the suppression is physical)."""
        auto = self._slow_grid().stack(self._slow_frames(), 1.0, self._slow_times())
        forced = self._slow_grid().stack(
            self._slow_frames(), 1.0, self._slow_times(),
            subtract_temporal_median=True,
        )
        assert forced.peak_snr < 0.5 * auto.peak_snr

    def test_auto_subtracts_for_fast_band(self) -> None:
        frames = _moving_source_frames(8.0, 10.0, 0.0, 1.0, seed=3)
        grid = PsiPhiStacker(
            np.arange(0.0, 21.0, 5.0), np.array([0.0]), psf_sigma_px=_PSF_SIGMA
        )
        result = grid.stack(frames, 1.0, _times())
        assert result.median_image is not None  # fast band: subtraction on
        assert result.best_vx_px_s == pytest.approx(10.0)


class TestSpeed:
    """Integer-shift ψ/φ search must beat interpolating shift-and-add."""

    def test_faster_than_classical_stacker(self) -> None:
        """Same data, same 169-hypothesis grid.  The ψ/φ path pays two
        small convolutions per frame once, then only slice-adds per
        hypothesis; the classical path interpolates every frame for every
        hypothesis, so the advantage grows with the hypothesis count
        (measured 4.4× total / ~7× marginal at 13×13 on 128² frames,
        larger on full frames).  Assert a conservative ≥2× so the test
        stays robust on slow CI hosts."""
        rng = np.random.default_rng(61)
        frames = [rng.normal(0.0, 1.0, (128, 128)) for _ in range(20)]
        times = [(k - 9.5) * 0.2 for k in range(20)]
        grid = np.linspace(-20.0, 20.0, 13)  # 13×13 hypotheses

        t0 = time.perf_counter()
        PsiPhiStacker(grid, grid, psf_sigma_px=_PSF_SIGMA).stack(
            frames, 1.0, times, subtract_temporal_median=False
        )
        t_psi = time.perf_counter() - t0

        t0 = time.perf_counter()
        Stacker(grid, grid).stack(frames, 1.0, times)
        t_classic = time.perf_counter() - t0

        assert t_psi < t_classic / 2.0, (
            f"psi/phi {t_psi:.3f}s vs classical {t_classic:.3f}s"
        )


class TestPoissonTailThreshold:
    """Bernstein detection threshold (contract since 2026-07-28)."""

    def test_gaussian_limit_at_zero_bound_scale(self) -> None:
        """ε = 0 reproduces the former Gaussian extreme-value contract
        u* = √(2 ln N) + margin exactly."""
        for n in (2, 1000, 3_000_000):
            assert poisson_tail_threshold(n, 0.0, 0.5) == pytest.approx(
                math.sqrt(2.0 * math.log(n)) + 0.5
            )

    def test_inverts_the_bernstein_equation(self) -> None:
        """u* (margin 0) solves N·exp(−u²/(2(1 + εu/3))) = 1 — the
        expected-exceedances design point of the module docstring."""
        for n, eps in ((10_000, 0.1), (3_000_000, 0.19), (10**9, 0.5)):
            u = poisson_tail_threshold(n, eps)
            residual = n * math.exp(-u * u / (2.0 * (1.0 + eps * u / 3.0)))
            assert residual == pytest.approx(1.0, rel=1e-9)

    def test_monotone_in_trials_and_bound_scale(self) -> None:
        assert poisson_tail_threshold(10**6, 0.2) > poisson_tail_threshold(
            10**4, 0.2
        )
        assert poisson_tail_threshold(10**6, 0.2) > poisson_tail_threshold(
            10**6, 0.0
        )

    def test_margin_is_pure_offset(self) -> None:
        base = poisson_tail_threshold(10**6, 0.15)
        assert poisson_tail_threshold(10**6, 0.15, 0.5) == pytest.approx(
            base + 0.5
        )


class TestBoundScaleEpsilon:
    """ε = b/√Φ of the trajectory-summed statistic."""

    def test_scalar_noise_closed_form(self) -> None:
        """ε = (P_max/‖P‖₂)·s/(σ√N) for scalar σ over N frames."""
        kernel = gaussian_psf_kernel(_PSF_SIGMA)
        kappa = float(kernel.max()) / math.sqrt(float((kernel**2).sum()))
        for sigma, n, s in ((1.0, 12, 1.0), (2.5, 25, 4.0)):
            got = bound_scale_epsilon(
                kernel, sigma, n_frames=n, quant_step_e=s
            )
            assert got == pytest.approx(
                kappa * s / (sigma * math.sqrt(n)), rel=1e-12
            )

    def test_per_frame_sigmas_use_worst_single_event_over_full_coverage(
        self,
    ) -> None:
        """b comes from the noisiest-weighted frame (max P_max·s/σ_k²);
        Φ is the full-coverage sum — matches the hand-built expression."""
        kernel = gaussian_psf_kernel(_PSF_SIGMA)
        p_max = float(kernel.max())
        p_sq = float((kernel**2).sum())
        sigmas = [0.8, 1.0, 2.0]
        got = bound_scale_epsilon(kernel, sigmas, quant_step_e=1.0)
        b = max(p_max / (s * s) for s in sigmas)
        phi = sum(p_sq / (s * s) for s in sigmas)
        assert got == pytest.approx(b / math.sqrt(phi), rel=1e-12)

    def test_shrinks_with_frames_and_counts(self) -> None:
        """CLT direction: more frames or higher counts (σ ∝ √λ at fixed
        s) drive ε → 0 — the Gaussian regime recovers itself."""
        kernel = gaussian_psf_kernel(_PSF_SIGMA)
        e12 = bound_scale_epsilon(kernel, 1.0, n_frames=12)
        e48 = bound_scale_epsilon(kernel, 1.0, n_frames=48)
        assert e48 == pytest.approx(e12 / 2.0, rel=1e-12)
        assert bound_scale_epsilon(kernel, 5.0, n_frames=12) < e12

    def test_input_validation(self) -> None:
        kernel = gaussian_psf_kernel(_PSF_SIGMA)
        with pytest.raises(ValueError):
            bound_scale_epsilon(kernel, 1.0)  # scalar σ needs n_frames
        with pytest.raises(ValueError):
            bound_scale_epsilon(kernel, [1.0, -1.0])
        with pytest.raises(ValueError):
            bound_scale_epsilon(kernel, 1.0, n_frames=5, quant_step_e=0.0)


class TestQuantisationStep:
    """Recorded-value ladder estimator (bound-scale input)."""

    def test_continuous_noise_floors_at_one_electron(self) -> None:
        """Continuous-valued frames carry no resolvable ladder: s = the
        1 e⁻ photo-electron floor regardless of σ."""
        rng = np.random.default_rng(3)
        for sigma in (0.5, 3.0, 30.0):
            frames = [rng.normal(0.0, sigma, (80, 80)) for _ in range(5)]
            assert quantisation_step_e(frames) == 1.0

    def test_coarse_ladder_detected(self) -> None:
        """8-bit-video-like data (low counts, gain ≫ 1 e⁻/ADU) measures
        the ladder step, not the floor."""
        rng = np.random.default_rng(4)
        gain = 4.0
        frames = [
            gain * rng.poisson(2.0, (80, 80)).astype(np.float64)
            for _ in range(5)
        ]
        assert quantisation_step_e(frames) == pytest.approx(gain, rel=1e-6)

    def test_integer_poisson_at_lambda_25_keeps_the_floor(self) -> None:
        """λ = 25 integer-quantised frames: consecutive-frame ties are
        rare (≈ 1/√(4πλ) ≈ 5.6 %), so no ladder is claimed and s stays at
        the 1 e⁻ floor — the honest value, since the Poisson jump IS one
        electron there.  (The −11 % MAD-discreteness σ bias at this λ is
        a property of the σ estimator and is deliberately not 'corrected'
        here — repo-wide caution.)"""
        rng = np.random.default_rng(5)
        frames = [
            rng.poisson(25.0, (80, 80)).astype(np.float64) for _ in range(5)
        ]
        assert quantisation_step_e(frames) == 1.0

    def test_single_frame_and_constant_frames_floor(self) -> None:
        assert quantisation_step_e([np.zeros((8, 8))]) == 1.0
        frames = [np.full((8, 8), 7.0) for _ in range(4)]
        assert quantisation_step_e(frames) == 1.0
