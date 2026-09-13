"""Tests for the analytic sensitivity model (opta_pipeline.sensitivity).

Pure math — no frame generation.  Pins the equivalences that keep the
detection-vs-duration figure (fig_32) tied to the thresholds the pipeline
actually applies:

* u_star_blind(T) delegates to coarse_fine._u_star_fine at the
  fine-equivalent step PSF_fwhm / T;
* the u* floor exceeds the OpTA.NOD.DET 5σ acceptance level at the CI
  completeness geometry (the calibrated tension documented in
  docs/pipeline-stack-assessment.md F6);
* the √T / Pogson scaling identities of snr_full_pass;
* seed_window_auto_s mirrors the auto rule in blind_coarse_fine_search.

Requirement traceability: OpTA.NOD.DET.
"""

from __future__ import annotations

import math

import pytest

from opta_pipeline.coarse_fine import _u_star_fine
from opta_pipeline.sensitivity import (
    PSF_FWHM_FACTOR,
    blind_limit,
    crossing_time_s,
    seed_window_auto_s,
    seeding_floor,
    snr_full_pass,
    u_star_blind,
    u_star_prior,
)

# CI completeness-scene geometry (test_validation_completeness.py /
# fig_22): SENSOR_SMALL 400×300 at 25 fps, ±10 px/s search box.
_H, _W = 300, 400
_V_MAX = 10.0
_PSF_SIGMA = 1.5
_MARGIN = 0.5


class TestUStarBlind:
    def test_matches_coarse_fine_formula_exactly(self):
        """u_star_blind must be _u_star_fine at step = PSF_fwhm / T."""
        for t_s in [0.5, 2.0, 11.0, 39.0]:
            fine_step = PSF_FWHM_FACTOR * _PSF_SIGMA / t_s
            expected = _u_star_fine(_H, _W, _V_MAX, fine_step, _MARGIN)
            got = u_star_blind(
                t_s,
                height_px=_H,
                width_px=_W,
                v_max_px_s=_V_MAX,
                psf_sigma_px=_PSF_SIGMA,
                far_margin_sigma=_MARGIN,
            )
            assert got == expected

    def test_monotone_non_decreasing_in_duration(self):
        """Longer passes refine the velocity grid → u* can only grow."""
        ts = [0.5, 1.0, 2.0, 5.0, 11.0, 20.0, 39.0, 60.0]
        vals = [
            u_star_blind(t, height_px=_H, width_px=_W, v_max_px_s=_V_MAX)
            for t in ts
        ]
        assert all(b >= a for a, b in zip(vals, vals[1:]))

    def test_ci_geometry_floor_exceeds_nod_det_5_sigma(self):
        """At the 2 s CI scene the calibrated floor sits above NOD.DET ≥ 5σ.

        This is the documented tension: blind 2 s windows cannot accept at
        the requirement's 5σ level without breaking the false-alarm
        guarantee.
        """
        u2 = u_star_blind(2.0, height_px=_H, width_px=_W, v_max_px_s=_V_MAX)
        assert u2 > 5.0

    def test_rejects_non_positive_duration(self):
        with pytest.raises(ValueError):
            u_star_blind(0.0, height_px=_H, width_px=_W, v_max_px_s=_V_MAX)


class TestSnrFullPass:
    def test_anchor_identity(self):
        assert snr_full_pass(2.0, snr_ref=4.7, t_ref_s=2.0) == 4.7

    def test_sqrt_t_scaling(self):
        base = snr_full_pass(2.0, snr_ref=4.7, t_ref_s=2.0)
        assert snr_full_pass(8.0, snr_ref=4.7, t_ref_s=2.0) == pytest.approx(
            2.0 * base
        )

    def test_pogson_magnitude_scaling(self):
        bright = snr_full_pass(
            2.0, snr_ref=4.7, t_ref_s=2.0, mag=12.0, mag_ref=13.0
        )
        assert bright == pytest.approx(4.7 * 10.0 ** 0.4)

    def test_mag_without_mag_ref_raises(self):
        with pytest.raises(ValueError):
            snr_full_pass(2.0, snr_ref=4.7, t_ref_s=2.0, mag=12.0)


class TestSeeding:
    def test_auto_window_mirrors_coarse_fine_rule(self):
        """T₀ = max(PSF_fwhm / coarse_step, 4 frames) — coarse_fine auto rule."""
        # PSF-limited branch: 3.5325/5.0 = 0.7065 > 4/25.
        assert seed_window_auto_s(
            coarse_step_px_s=5.0, psf_sigma_px=1.5, fps=25.0
        ) == pytest.approx(PSF_FWHM_FACTOR * 1.5 / 5.0)
        # Frame-count-limited branch: 4 frames at 2 fps = 2.0 s.
        assert seed_window_auto_s(
            coarse_step_px_s=50.0, psf_sigma_px=1.5, fps=2.0
        ) == pytest.approx(2.0)

    def test_floor_equals_prethreshold_at_one_window(self):
        assert seeding_floor(0.7, t0_s=0.7, prethreshold_sigma=4.0) == 4.0

    def test_floor_cancels_sqrt_t_gain(self):
        """Seeding floor grows as √T — same exponent as the signal gain."""
        f1 = seeding_floor(2.0, t0_s=0.5)
        f2 = seeding_floor(8.0, t0_s=0.5)
        assert f2 == pytest.approx(2.0 * f1)

    def test_blind_limit_is_max_of_parts(self):
        for t_s in [1.0, 5.0, 20.0]:
            lim = blind_limit(
                t_s,
                height_px=_H,
                width_px=_W,
                v_max_px_s=_V_MAX,
                t0_s=0.7,
            )
            u = u_star_blind(t_s, height_px=_H, width_px=_W, v_max_px_s=_V_MAX)
            s = seeding_floor(t_s, t0_s=0.7)
            assert lim == max(u, s)


class TestPriorAndCrossing:
    def test_prior_floor_matches_pipeline_gate_formula(self):
        """u* = √(2 ln(N_pix · N_hyp)) + margin — pipeline._stack_and_detect."""
        n_hyp = 25
        expected = math.sqrt(2.0 * math.log(_H * _W * n_hyp)) + _MARGIN
        assert u_star_prior(
            height_px=_H, width_px=_W, n_hyp=n_hyp, far_margin_sigma=_MARGIN
        ) == pytest.approx(expected)

    def test_prior_floor_below_blind_floor(self):
        """A small cued box always beats the blind fine-equivalent grid."""
        prior = u_star_prior(height_px=_H, width_px=_W, n_hyp=25)
        blind = u_star_blind(11.0, height_px=_H, width_px=_W, v_max_px_s=_V_MAX)
        assert prior < blind

    def test_crossing_time_roundtrip(self):
        t = crossing_time_s(snr_ref=4.7, t_ref_s=2.0, threshold=6.0)
        assert snr_full_pass(t, snr_ref=4.7, t_ref_s=2.0) == pytest.approx(6.0)

    def test_crossing_time_with_magnitude_offset(self):
        t = crossing_time_s(
            snr_ref=4.7, t_ref_s=2.0, threshold=6.0, mag=13.5, mag_ref=13.0
        )
        assert snr_full_pass(
            t, snr_ref=4.7, t_ref_s=2.0, mag=13.5, mag_ref=13.0
        ) == pytest.approx(6.0)


class TestBernsteinContract:
    """2026-07-28 contract: ε threading + depth-matched auto T₀."""

    def test_u_star_blind_threads_bound_scale_to_pipeline_formula(self):
        """With ε > 0 the delegation to coarse_fine._u_star_fine holds
        exactly — the analytic curve cannot drift from the applied gate."""
        for t_s, eps in [(2.0, 0.094), (11.0, 0.04)]:
            fine_step = PSF_FWHM_FACTOR * _PSF_SIGMA / t_s
            expected = _u_star_fine(_H, _W, _V_MAX, fine_step, _MARGIN, eps)
            got = u_star_blind(
                t_s,
                height_px=_H,
                width_px=_W,
                v_max_px_s=_V_MAX,
                psf_sigma_px=_PSF_SIGMA,
                far_margin_sigma=_MARGIN,
                bound_scale_eps=eps,
            )
            assert got == expected
            assert got > u_star_blind(
                t_s,
                height_px=_H,
                width_px=_W,
                v_max_px_s=_V_MAX,
                psf_sigma_px=_PSF_SIGMA,
                far_margin_sigma=_MARGIN,
            )

    def test_bound_scale_matches_likelihood_estimator(self):
        """sensitivity.bound_scale mirrors likelihood.bound_scale_epsilon
        with the same kernel construction (scalar-σ closed form)."""
        from opta_pipeline.likelihood import (
            bound_scale_epsilon,
            gaussian_psf_kernel,
        )
        from opta_pipeline.sensitivity import bound_scale

        got = bound_scale(psf_sigma_px=_PSF_SIGMA, noise_rms=1.0, n_frames=50)
        want = bound_scale_epsilon(
            gaussian_psf_kernel(_PSF_SIGMA), 1.0, n_frames=50
        )
        assert got == want

    def test_auto_window_depth_term_inverts_seeding_floor(self):
        """T₀ ⊇ T·(pre/u*)² ⇒ seeding_floor(T, T₀) ≤ u*(T), by construction."""
        for t_s in [2.0, 11.0, 39.0]:
            u = u_star_blind(
                t_s, height_px=_H, width_px=_W, v_max_px_s=_V_MAX,
                bound_scale_eps=0.05,
            )
            t0 = seed_window_auto_s(
                coarse_step_px_s=5.0, psf_sigma_px=1.5, fps=25.0,
                t_span_s=t_s, u_star=u,
            )
            assert seeding_floor(t_s, t0_s=t0) <= u + 1e-9

    def test_auto_window_without_depth_args_is_legacy_rule(self):
        """Omitting (t_span_s, u_star) reproduces the depth-agnostic rule."""
        legacy = max(PSF_FWHM_FACTOR * 1.5 / 5.0, 4.0 / 25.0)
        assert seed_window_auto_s(
            coarse_step_px_s=5.0, psf_sigma_px=1.5, fps=25.0
        ) == pytest.approx(legacy)
