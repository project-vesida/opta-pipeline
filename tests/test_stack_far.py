"""Noise-only false-alarm behaviour of the blind stack search (F6).

Blind track-and-stack maximises over N_pix pixels × N_hyp velocity
hypotheses, so its noise floor is an extreme-value statistic,

    u* ≈ √(2 ln(N_pix · N_hyp)),

not the single-frame detection threshold.  These tests measure the
pure-noise behaviour at a small, exactly-analysable scale and pin the
consequences:

* ``StackResult.peak_snr`` on pure noise lands at ≈ 4σ here — ABOVE the
  3.0 config detection threshold, and at full geometry (1920×1080,
  41×41 hypotheses) u* ≈ 6.6σ — above the NOD.DET ≥ 5 acceptance value.
  ``peak_snr`` therefore cannot be gated at per-frame thresholds; this is
  the statistical face of the "stacked-SNR can mint fake tracklets" issue
  (docs/endgame-plan.md, docs/pipeline-stack-assessment.md F6).
* The connected-component detector (min_pixels = 5) suppresses
  white-noise false alarms at this scale.  That is a floor, not a safety
  claim: real frames add non-white structure (star wings, background
  residuals, hot pixels) that only field data can quantify.

Geometry note: the velocity grid (multiples of 20 px/s) and frame times
(multiples of 0.25 s) are chosen so every shift is an integer number of
pixels — no interpolation, so white-noise statistics apply exactly.

Reproduce the full FAR-vs-threshold curve:
    python3 opta-pipeline/scripts/stacking_far_mc.py
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from opta_pipeline.detect import detect_sources
from opta_pipeline.stack import Stacker

_H, _W = 96, 96
_N_FRAMES = 20
_SEEDS = (0, 1, 2, 3)

_SNR_THRESHOLD_CONFIG = 3.0  # pipeline_defaults.yaml detection.snr_threshold
_NOD_DET_MIN_SNR = 5.0  # OpTA.NOD.DET acceptance (see test_stack.py)


def _integer_shift_stacker() -> Stacker:
    """11×11 grid whose shifts are all integer pixels (see module note)."""
    v = np.arange(-100.0, 101.0, 20.0)
    return Stacker(vx_grid=v, vy_grid=v)


def _times_s() -> list[float]:
    return [(k - (_N_FRAMES - 1) / 2.0) * 0.25 for k in range(_N_FRAMES)]


def _noise_stack(seed: int):
    rng = np.random.default_rng(seed)
    frames = [rng.normal(0.0, 1.0, (_H, _W)) for _ in range(_N_FRAMES)]
    return _integer_shift_stacker().stack(frames, 1.0, _times_s())


def _extreme_value_bound(n_pix: int, n_hyp: int) -> float:
    """Expected pure-noise maximum over n_pix·n_hyp Gaussian trials."""
    return math.sqrt(2.0 * math.log(n_pix * n_hyp))


class TestNoiseOnlyPeakSnr:
    """peak_snr on pure noise: above config threshold, below u* bound."""

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_exceeds_config_threshold(self, seed: int) -> None:
        """F6: pure noise yields peak_snr ≈ 4σ (measured 3.78–4.24 over
        these seeds) — the 3.0 config threshold cannot gate blind stacks."""
        result = _noise_stack(seed)
        assert result.peak_snr > _SNR_THRESHOLD_CONFIG

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_below_extreme_value_bound(self, seed: int) -> None:
        """The analytic u* bound holds → usable as a calibrated gate."""
        result = _noise_stack(seed)
        n_hyp = 11 * 11
        u_star = _extreme_value_bound(_H * _W, n_hyp)
        assert result.peak_snr < u_star + 0.5

    def test_full_geometry_bound_exceeds_nod_det_acceptance(self) -> None:
        """At full geometry the noise floor is ABOVE the NOD.DET value.

        1920×1080 pixels × 41² hypotheses → u* ≈ 6.6σ > 5.0.  A blind-mode
        'peak SNR ≥ 5' criterion is therefore satisfiable by pure noise;
        acceptance must use either the component detector with a
        calibrated threshold or a trials-corrected statistic.
        """
        u_star_full = _extreme_value_bound(1920 * 1080, 41 * 41)
        assert u_star_full > _NOD_DET_MIN_SNR


class TestNoiseOnlyComponentDetector:
    """Connected-component detector on the best noise stack."""

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_min_pixels_suppresses_white_noise(self, seed: int) -> None:
        """Characterization: min_pixels=5 clustering yields zero false
        components in white noise at this scale.  NOT a safety claim for
        real frames (non-white structure); field FAR remains open (F6)."""
        result = _noise_stack(seed)
        dets = detect_sources(
            result.stacked,
            noise_rms=result.noise_rms,
            snr_threshold=_SNR_THRESHOLD_CONFIG,
            min_pixels=5,
            max_pixels=500,
            elongation_threshold=2.0,
        )
        assert len(dets) == 0

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_calibrated_threshold_rejects_single_pixels(self, seed: int) -> None:
        """Even with min_pixels=1, thresholding at u* rejects pure noise."""
        result = _noise_stack(seed)
        u_star = _extreme_value_bound(_H * _W, 11 * 11)
        dets = detect_sources(
            result.stacked,
            noise_rms=result.noise_rms,
            snr_threshold=u_star,
            min_pixels=1,
            max_pixels=500,
            elongation_threshold=2.0,
        )
        assert len(dets) == 0


class TestPoissonTailContract:
    """Blank-scene FAR regression for the Poisson-aware u* (2026-07-28).

    The Gaussian extreme-value gate measurably under-covered the
    sub-exponential tail of low-count Poisson noise: on blank SMOKE-level
    scenes (~0.3 e⁻/px sky, seeds 0–99, blessed pins) the honest
    fine-grid peak exceeded u* = 5.96 in 9/100 scenes (Gaussian control
    1/100), and the shipped coarse-fine path emitted false tracklets in
    6/100 (recipe in the likelihood module docstring; deterministic PCG64
    draws). This
    class pins the replacement contract on uint8-style quantised
    low-count Poisson blanks through the stacked search path:

    * zero candidates out of the blind coarse-fine search;
    * the applied threshold IS poisson_tail_threshold at the ε estimated
      from the searched frames — and strictly dominates the old Gaussian
      bound (the raise is the fix, not margin luck);
    * a coarser recorded ladder (gain ≫ 1 e⁻/ADU, the 8-bit-video case)
      raises ε — the gate tracks measured quantisation.
    """

    _SH, _SW = 150, 200
    _N_FRAMES2 = 12
    _FPS = 25.0
    _PSF_SIGMA = 0.849  # production stacking.psf_sigma_px
    _V_MAX = 10.0
    _COARSE = 5.0
    _LAM_E = 3.0  # low-count sky, electrons/px

    def _times2(self) -> list[float]:
        n = self._N_FRAMES2
        return [(k - (n - 1) / 2.0) / self._FPS for k in range(n)]

    def _blank_scene(self, seed: int, gain_e_adu: float):
        """Quantised blank sky: Poisson events of size ``gain`` e⁻.

        Honest per-frame handling mirrors the pipeline: subtract the
        frame mean (background), estimate σ from the frame itself with
        the ≥ 1 e⁻ clamp.  Nothing is tuned to any detection outcome —
        the scene is blank by construction (design rule 10).
        """
        rng = np.random.default_rng(seed)
        frames = [
            gain_e_adu
            * rng.poisson(
                self._LAM_E / gain_e_adu, (self._SH, self._SW)
            ).astype(np.float64)
            for _ in range(self._N_FRAMES2)
        ]
        cal = [f - f.mean() for f in frames]
        sigmas = [max(float(f.std()), 1.0) for f in cal]
        return cal, sigmas

    def _search(self, cal, sigmas):
        from opta_pipeline.coarse_fine import blind_coarse_fine_search

        return blind_coarse_fine_search(
            cal,
            sigmas,
            self._times2(),
            v_max_px_s=self._V_MAX,
            coarse_step_px_s=self._COARSE,
            psf_sigma_px=self._PSF_SIGMA,
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
    def test_quantised_blank_emits_no_candidates(self, seed: int) -> None:
        cal, sigmas = self._blank_scene(seed, gain_e_adu=4.0)
        assert self._search(cal, sigmas).candidates == ()

    def test_gate_is_bernstein_at_measured_epsilon(self) -> None:
        """u* equals poisson_tail_threshold at the frames' own measured
        bound scale (ladder step 4 e⁻ detected, σ from the data) and
        strictly dominates the former Gaussian extreme-value bound."""
        from opta_pipeline.likelihood import (
            bound_scale_epsilon,
            gaussian_psf_kernel,
            poisson_tail_threshold,
            quantisation_step_e,
        )

        cal, sigmas = self._blank_scene(0, gain_e_adu=4.0)
        r = self._search(cal, sigmas)
        step = quantisation_step_e(cal)
        # Ladder detected (not the 1 e⁻ floor); the small deficit vs the
        # true 4.0 is the per-frame mean-subtraction offset between
        # consecutive frames — the documented smooth-background drift.
        assert step == pytest.approx(4.0, rel=0.05)
        eps = bound_scale_epsilon(
            gaussian_psf_kernel(self._PSF_SIGMA), sigmas, quant_step_e=step
        )
        n_axis = int(2.0 * self._V_MAX / r.fine_step_px_s) + 1
        n_trials = self._SH * self._SW * n_axis * n_axis
        assert r.u_star == pytest.approx(
            poisson_tail_threshold(n_trials, eps, 0.5), rel=1e-9
        )
        gaussian_u = math.sqrt(2.0 * math.log(n_trials)) + 0.5
        assert r.u_star > gaussian_u + 0.5  # the raise is structural

    def test_coarser_ladder_raises_the_gate(self) -> None:
        """Same sky, coarser recorded quantisation ⇒ larger ε ⇒ higher
        u* — the contract tracks the data's own discreteness."""
        cal1, sig1 = self._blank_scene(1, gain_e_adu=1.0)
        cal4, sig4 = self._blank_scene(1, gain_e_adu=4.0)
        assert self._search(cal4, sig4).u_star > self._search(cal1, sig1).u_star
