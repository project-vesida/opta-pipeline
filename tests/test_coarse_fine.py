"""Tests for the coarse-to-fine blind velocity search (Phase 2, F1/F2/F7).

The F1 spec that was a strict xfail in test_stack_offnode.py flips here:
the worst-case off-node target — which the criterion-violating coarse
grid loses 16× — is recovered by the two-stage search at full SNR.

The sensitivity trade of multi-stage track-before-detect is characterized
explicitly (not hidden): stage-1 seeding needs ≈ prethreshold_sigma per
window, so the effective limiting full-pass SNR is
max(u*, prethreshold · √(N/N_window)) — both sides of that boundary are
pinned below.
"""

from __future__ import annotations

import numpy as np
import pytest

from opta_pipeline.coarse_fine import blind_coarse_fine_search
from opta_pipeline.likelihood import PsiPhiStacker, gaussian_psf_kernel

_H, _W = 128, 128
_N = 25
_PSF_SIGMA = 1.2
_TS = [(k - (_N - 1) / 2.0) * 0.2 for k in range(_N)]  # ±2.4 s
_V_MAX = 40.0
_COARSE_STEP = 20.0  # deliberately criterion-violating for the full pass


def _search(frames, **kw):
    defaults = dict(
        v_max_px_s=_V_MAX,
        coarse_step_px_s=_COARSE_STEP,
        psf_sigma_px=_PSF_SIGMA,
        subtract_median=False,
    )
    defaults.update(kw)
    return blind_coarse_fine_search(frames, 1.0, _TS, **defaults)


def _frames(sources, noise, seed):
    """sources: list of (y0, vx, amplitude); trajectories through x=64."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:_H, 0:_W].astype(np.float64)
    frames = []
    for t in _TS:
        f = rng.normal(0.0, noise, (_H, _W)) if noise > 0 else np.zeros((_H, _W))
        for y0, vx, amp in sources:
            xc = _W / 2.0 + vx * t
            f += amp * np.exp(
                -(((xx - xc) ** 2 + (yy - y0) ** 2) / (2.0 * _PSF_SIGMA**2))
            )
        frames.append(f)
    return frames


def _matched_snr(amplitude: float, n_vis: int, noise: float) -> float:
    kernel = gaussian_psf_kernel(_PSF_SIGMA)
    flux = amplitude * 2.0 * np.pi * _PSF_SIGMA**2
    return flux * float(np.sqrt((kernel * kernel).sum())) * np.sqrt(n_vis) / noise


#: The canonical off-node scene: A = 8, v = (10, 0) px/s, noise seed 1.
#: Three test classes below feed *byte-identical* frames through ``_search``
#: with the *same* default kwargs, so the identical blind search used to run
#: eight times per session (once here, six times as the ``want`` reference of
#: TestNoiseRmsScalarTypes, once in TestGateHonesty).  ``_frames`` is a pure
#: function of (sources, noise, seed) via ``default_rng``, and neither
#: ``blind_coarse_fine_search`` nor ``make_psi_phi`` writes into its inputs
#: (both go through ``np.asarray``/``np.stack`` copies), so one module-scoped
#: evaluation is exactly the value each call site computed for itself.
@pytest.fixture(scope="module")
def offnode_frames() -> list[np.ndarray]:
    return _frames([(64.0, 10.0, 8.0)], 1.0, 1)


@pytest.fixture(scope="module")
def offnode_result(offnode_frames: list[np.ndarray]):
    return _search(offnode_frames)


class TestOffNodeSpecFlipped:
    """The F1 spec (was strict xfail in test_stack_offnode.py)."""

    def test_worst_case_off_node_recovered_at_full_snr(
        self, offnode_result
    ) -> None:
        """v = 10 px/s (midpoint between 20 px/s nodes): the coarse grid
        alone keeps 6.3 % of the peak; the two-stage search recovers the
        target at ≥ 90 % of the on-node SNR, with the velocity resolved
        to the full-pass criterion step."""
        r_off = offnode_result
        r_on = _search(_frames([(64.0, 20.0, 8.0)], 1.0, 1))
        assert len(r_off.candidates) == 1 and len(r_on.candidates) == 1
        c_off, c_on = r_off.candidates[0], r_on.candidates[0]
        assert c_off.vx_px_s == pytest.approx(10.0, abs=r_off.fine_step_px_s)
        assert c_off.snr >= 0.90 * c_on.snr
        # And both sit at the analytic matched-filter prediction.
        assert c_off.snr == pytest.approx(_matched_snr(8.0, _N, 1.0), rel=0.2)

    def test_ghost_candidates_suppressed(self) -> None:
        """Trail-crossing side hypotheses (measured ~10× lower SNR) are
        removed by trajectory-space NMS — exactly one candidate."""
        r = _search(_frames([(64.0, 10.0, 20.0)], 1.0, 2))
        assert len(r.candidates) == 1


class TestNoiseRmsScalarTypes:
    """NumPy scalar σ must reach the scalar branch at this entry point too.

    ``blind_coarse_fine_search`` and ``make_psi_phi`` both accept "scalar or
    per-frame" ``noise_rms``; they used two *different* predicates, and this
    one missed every NumPy scalar — an ``np.float32`` σ (a perfectly ordinary
    ``calibrate``/``detect`` product) fell through to ``noise_rms[i]`` on a
    0-d value.  One shared predicate, tested from both sides.
    """

    @pytest.mark.parametrize(
        "scalar",
        [
            np.float64(1.0),
            np.float32(1.0),
            np.int64(1),
            np.int32(1),
            np.asarray(1.0),  # 0-d array: np.mean(...) / arr[()] shape
            np.asarray(1, dtype=np.int64),
        ],
        ids=[
            "float64", "float32", "int64", "int32",
            "0d-float-array", "0d-int-array",
        ],
    )
    def test_numpy_scalar_noise_matches_python_float(
        self, scalar, offnode_frames, offnode_result
    ) -> None:
        frames = offnode_frames
        got = blind_coarse_fine_search(
            frames,
            scalar,
            _TS,
            v_max_px_s=_V_MAX,
            coarse_step_px_s=_COARSE_STEP,
            psf_sigma_px=_PSF_SIGMA,
            subtract_median=False,
        )
        want = offnode_result  # same frames, same kwargs — see the fixture
        assert len(got.candidates) == len(want.candidates) == 1
        assert got.candidates[0].snr == pytest.approx(want.candidates[0].snr)


class TestFalseAlarms:
    """The u* gate survives the two-stage architecture."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_noise_only_yields_no_candidates(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        frames = [rng.normal(0.0, 1.0, (_H, _W)) for _ in range(_N)]
        r = _search(frames)
        assert r.candidates == ()

    def test_u_star_uses_fine_equivalent_trials(self) -> None:
        """The final threshold is computed against the fine-equivalent
        full grid — conservative w.r.t. the hypotheses actually visited —
        with the Bernstein bound scale ε estimated from the searched
        frames (Poisson-aware contract, 2026-07-28).  These frames are
        continuous unit-σ Gaussian noise, so quantisation_step_e floors
        at the 1 e⁻ photo-electron quantum and
        ε = (P_max/‖P‖₂)·1/(σ√N) = κ(σ_psf=1.2)/√25 ≈ 0.094; with
        L = ln(N_pix·n_axis²) the gate is Lε/3 + √((Lε/3)² + 2L) + 0.5
        ≈ 7.39 (was the Gaussian √(2L) + 0.5 ≈ 6.75)."""
        from opta_pipeline.likelihood import (
            bound_scale_epsilon,
            poisson_tail_threshold,
        )

        r = _search(_frames([(64.0, 10.0, 8.0)], 1.0, 3))
        n_axis = int(2 * _V_MAX / r.fine_step_px_s) + 1
        eps = bound_scale_epsilon(
            gaussian_psf_kernel(_PSF_SIGMA), 1.0, n_frames=_N, quant_step_e=1.0
        )
        expected = poisson_tail_threshold(_H * _W * n_axis**2, eps, 0.5)
        assert r.u_star == pytest.approx(expected, rel=1e-6)
        # The Poisson-aware gate strictly dominates the former Gaussian
        # extreme-value bound (ε > 0 here).
        gaussian = np.sqrt(2 * np.log(_H * _W * n_axis**2)) + 0.5
        assert r.u_star > gaussian


class TestGateHonesty:
    """The final u* gate scores candidates on the honest full-pass statistic.

    Root-caused 2026-07-10 (blank-sweep FP triage): the pyramid's ROI
    search metric embeds (a) maximization over its own per-frame
    integer-rounding realizations and (b) a locally-biased scalar-median
    pedestal estimate.  Gating on it emitted blank-scene tracklets whose
    honest full-pass score was below u* (measured +1.9σ inflation on a
    blank SMOKE scene: reported 7.10 vs honest 5.20 at the same
    trajectory, u* = 5.96).  The refinement could also wander outside the
    ±v_max box the u* trials budget is derived for (seed ± half_width
    reaches v_max + half_width; both observed false positives sat outside
    the box).
    """

    def test_candidate_velocities_confined_to_search_box(self) -> None:
        """A bright mover OUTSIDE ±v_max must not drag candidate
        velocities out of the contracted search box (pre-fix: a vx = 55
        mover was reported at vx ≈ 48.5 > v_max = 40)."""
        r = _search(_frames([(64.0, 55.0, 8.0)], 1.0, 5))
        assert r.candidates  # corner seeds do fire on the out-of-box mover
        for c in r.candidates:
            assert abs(c.vx_px_s) <= _V_MAX + 1e-9
            assert abs(c.vy_px_s) <= _V_MAX + 1e-9

    def test_accepted_snr_is_honest_full_pass_score(
        self, offnode_frames, offnode_result
    ) -> None:
        """``candidate.snr`` equals the full-pass trajectory score on the
        same ψ/φ with the full-map median as zero level — the statistic
        the u* trials budget describes.  Pre-fix the last pyramid level's
        ROI metric was reported instead (82.21 vs honest 84.18 here; on
        faint blank-scene blobs the bias runs the other way and lifts
        sub-threshold noise over u*)."""
        from opta_pipeline.coarse_fine import _score_trajectory
        from opta_pipeline.likelihood import make_psi_phi

        frames = offnode_frames
        r = offnode_result
        assert r.candidates
        c = r.candidates[0]
        psi, phi = make_psi_phi(
            [np.asarray(f) for f in frames], 1.0,
            gaussian_psf_kernel(_PSF_SIGMA),
        )
        scored = _score_trajectory(
            psi, phi, np.asarray(_TS), c.x_px, c.y_px, c.vx_px_s, c.vy_px_s
        )
        assert scored is not None
        assert c.snr == pytest.approx(scored[0], abs=1e-9)


class TestMultiObject:
    """F7: every accepted candidate is returned with its own velocity."""

    def test_two_movers_both_recovered(self) -> None:
        r = _search(
            _frames([(40.0, 11.0, 6.0), (90.0, -7.0, 6.0)], 1.0, 7)
        )
        assert len(r.candidates) == 2
        vxs = sorted(c.vx_px_s for c in r.candidates)
        assert vxs[0] == pytest.approx(-7.0, abs=3 * r.fine_step_px_s)
        assert vxs[1] == pytest.approx(11.0, abs=3 * r.fine_step_px_s)
        ys = sorted(round(c.y_px) for c in r.candidates)
        assert ys == [40, 90]


class TestSensitivityTrade:
    """The multi-stage depth loss, characterized honestly.

    Stage-1 seeding is an OR over windows: a candidate is seeded if ANY
    window fluctuates above the pre-threshold (against the window map's
    own noise maximum ≈ √(2 ln N_pix) ≈ 4.2σ), so completeness near the
    limit is *probabilistic* — P ≈ 1 − (1 − Φc(u₁ − s/√M))^M.  Measured
    over 8 fixed noise seeds at this geometry with the auto 4-frame seed
    windows (recomputed 2026-07-02 via python3 -m pytest
    opta-pipeline/tests/test_coarse_fine.py): full-pass SNR 7.4 → 1/8
    recovered; SNR 12.8 → 8/8.  The blind completeness curve for the
    production completeness analysis must use this scheme (seed_window_s is the
    depth/compute knob), and cued (from_prior) mode keeps the full
    u*-limited depth — consistent with the 2026-07-02 decision that blind
    mode carries its own (shallower) completeness claims.
    """

    def test_bright_source_recovered_deterministically(self) -> None:
        # A=5: per-window matched SNR ≈ 21 with 4-frame windows.
        r = _search(_frames([(64.0, 10.0, 5.0)], 1.0, 11))
        assert len(r.candidates) == 1

    def test_completeness_probabilistic_near_the_limit(self) -> None:
        """Fixed-seed pin of the measured rolloff (see class docstring)."""
        def rate(amp: float) -> int:
            return sum(
                1
                for seed in range(8)
                if _search(_frames([(64.0, 10.0, amp)], 1.0, 100 + seed)).candidates
            )

        # Above the applied gate (Poisson-aware u* ≈ 7.39 at this geometry;
        # matched SNR 7.44, recomputed 2026-07-28), yet seeding-limited:
        probe = _search(_frames([(64.0, 10.0, 0.7)], 1.0, 100))
        assert _matched_snr(0.7, _N, 1.0) > probe.u_star
        assert rate(0.7) <= 4  # measured 1/8 — the seeding loss is real
        assert rate(1.2) >= 7  # measured 8/8 — recovered once seedable

    def test_cued_fine_search_keeps_full_depth(self) -> None:
        """A source in the probabilistic gap is measured at full matched-
        filter SNR by the criterion-sampled prior search — the blind/cued
        depth gap is real and quantified."""
        frames = _frames([(64.0, 10.0, 0.7)], 1.0, 100)
        fine = PsiPhiStacker.from_prior(
            10.0, 0.0, half_width_px_s=2.0, step_px_s=0.5,
            psf_sigma_px=_PSF_SIGMA,
        )
        res = fine.stack(frames, 1.0, _TS, subtract_temporal_median=False)
        assert res.peak_snr > 5.5  # ≈ 7.4 predicted, minus noise scatter


class TestSeedBinning:
    """b×b binned seeding: same recoveries, renormalized statistic, faster.

    Refinement and the final u* gate stay full-resolution, so binning may
    only affect WHICH seeds exist — every accepted candidate is still
    measured at full resolution.
    """

    def test_binned_recovers_off_node_target(self) -> None:
        """The F1 spec-flip scenario passes identically at b=2."""
        frames = _frames([(64.0, 10.0, 8.0)], noise=1.0, seed=7)
        r1 = _search(frames, seed_binning=1)
        r2 = _search(frames, seed_binning=2)
        assert r1.candidates and r2.candidates
        c1, c2 = r1.candidates[0], r2.candidates[0]
        assert c2.vx_px_s == pytest.approx(c1.vx_px_s, abs=r1.fine_step_px_s)
        assert c2.snr == pytest.approx(c1.snr, rel=0.05)

    def test_binned_statistic_is_renormalized(self) -> None:
        """The kernel-autocorrelation correction restores ~N(0,1).

        Without it the naive binned SNR is over-dispersed by ≈1.9 at b=2,
        σ_psf=1.5 (measured 1.92), silently loosening prethreshold_sigma.
        """
        from opta_pipeline.coarse_fine import _bin_sum, _binned_snr_dispersion
        from opta_pipeline.likelihood import make_psi_phi

        kernel = gaussian_psf_kernel(_PSF_SIGMA)
        corr = _binned_snr_dispersion(kernel, 2)
        rng = np.random.default_rng(17)
        frames = [rng.normal(0.0, 1.0, (_H, _W)) for _ in range(10)]
        psi, phi = make_psi_phi(frames, 1.0, kernel)
        psi_b = sum(_bin_sum(m, 2) for m in psi)
        phi_b = sum(_bin_sum(m, 2) for m in phi)
        snr = (psi_b / np.sqrt(phi_b)) / corr
        assert abs(float(snr[5:-5, 5:-5].std()) - 1.0) < 0.08

    def test_noise_only_still_yields_no_candidates(self) -> None:
        rng = np.random.default_rng(23)
        frames = [rng.normal(0.0, 1.0, (_H, _W)) for _ in range(_N)]
        result = _search(frames, seed_binning=2)
        assert result.candidates == ()
