"""Off-node velocity injection: quantify velocity-grid undersampling (F1).

Every prior stack test injects targets whose velocity sits exactly on a grid
node, which hides the registration smear a real (arbitrary-velocity) target
suffers.  These tests inject at the worst-case off-node velocity — the
midpoint between two nodes — and measure what survives.

Physics
-------
For a grid of step Δv and a stack of duration T (midpass-centred times,
|t| ≤ T/2), the worst-case residual velocity error is Δv/2, so the target
drifts up to Δv·T/4 pixels from its registered position at the window edges,
tracing a path of length Δv·T/2.  The stacked peak survives only if that
path stays inside the PSF:

    Δv ≲ PSF_fwhm / T          (the ``Stacker.from_prior`` criterion)

At the config default Δv = 20 px/s with T = 5 s the path is 50 px — the
peak collapses (measured ≈ 10× loss below the on-node peak here).  A grid
sampled at the criterion (Δv ≈ 0.5 px/s) restores it; that grid is ~40×
denser per axis, which is finding F2 (see
docs/pipeline-stack-assessment.md).

Structure
---------
TestOffNodeSmear      — characterization: current coarse-grid behaviour
TestCriterionControl  — positive control: criterion-sampled grid recovers
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from opta_pipeline.stack import Stacker

# Test geometry: chosen so all trajectories stay in-frame and runtimes are
# sub-second, while preserving the dimensionless quantity that matters:
# smear path (Δv·T/2) ≫ PSF_fwhm for the coarse grid, ≪ for the fine grid.
_H, _W = 128, 128
_N_FRAMES = 25
_T_SPAN_S = 5.0  # window duration; midpass-centred times
_PSF_SIGMA_PX = 1.2  # FWHM ≈ 2.83 px
_AMPLITUDE = 100.0
_NOISE_RMS = 1.0  # nominal; frames are noise-free for determinism

_COARSE_STEP = 20.0  # pipeline_defaults.yaml stacking.velocity_step_px_s
# from_prior criterion: step ≈ PSF_fwhm / T = 2.83 / 5 ≈ 0.57 → round down
_FINE_STEP = 0.5

_V_ON_NODE = 20.0  # a node of the coarse grid
_V_OFF_NODE = 10.0  # worst case: midpoint between nodes 0 and 20


def _times_s() -> list[float]:
    """Midpass-centred frame times spanning [-T/2, +T/2]."""
    dt = _T_SPAN_S / (_N_FRAMES - 1)
    return [(k - (_N_FRAMES - 1) / 2.0) * dt for k in range(_N_FRAMES)]


def _gaussian_frames(vx_px_s: float, vy_px_s: float) -> list[np.ndarray]:
    """Render a Gaussian source moving at (vx, vy) through frame centre."""
    yy, xx = np.mgrid[0:_H, 0:_W].astype(np.float64)
    x0, y0 = _W / 2.0, _H / 2.0
    frames = []
    for t in _times_s():
        xc = x0 + vx_px_s * t
        yc = y0 + vy_px_s * t
        r2 = (xx - xc) ** 2 + (yy - yc) ** 2
        frames.append(_AMPLITUDE * np.exp(-r2 / (2.0 * _PSF_SIGMA_PX**2)))
    return frames


def _coarse_stacker() -> Stacker:
    """Config-default coarse grid bracketing both test velocities."""
    vx = np.arange(-40.0, 41.0, _COARSE_STEP)  # [-40, -20, 0, 20, 40]
    vy = np.array([0.0])
    return Stacker(vx_grid=vx, vy_grid=vy)


class TestOffNodeSmear:
    """Coarse grid: on-node targets stack, off-node targets smear away."""

    def test_on_node_target_stacks_coherently(self) -> None:
        """Sanity: a target ON a grid node retains ≈ N× single-frame peak."""
        frames = _gaussian_frames(_V_ON_NODE, 0.0)
        result = _coarse_stacker().stack(frames, _NOISE_RMS, _times_s())
        assert result.best_vx_px_s == pytest.approx(_V_ON_NODE)
        # Bilinear registration of a σ=1.2 px Gaussian loses only a few
        # percent of the peak per frame.
        assert result.stacked.max() > 0.90 * _N_FRAMES * _AMPLITUDE

    def test_off_node_target_smears(self) -> None:
        """Characterization (F1): worst-case off-node peak collapses ~10×.

        Residual Δv/2 = 10 px/s over |t| ≤ 2.5 s drags the source along a
        50 px path; the per-pixel dwell time — not noise — sets the peak.
        This test pins the CURRENT behaviour so the Phase 2 fix has a
        baseline to beat; the companion xfail below states the target.
        """
        frames_on = _gaussian_frames(_V_ON_NODE, 0.0)
        frames_off = _gaussian_frames(_V_OFF_NODE, 0.0)
        stacker = _coarse_stacker()
        peak_on = stacker.stack(frames_on, _NOISE_RMS, _times_s()).stacked.max()
        peak_off = stacker.stack(frames_off, _NOISE_RMS, _times_s()).stacked.max()

        ratio = peak_off / peak_on
        assert ratio < 0.35, (
            f"Off-node/on-node peak ratio {ratio:.3f} unexpectedly high — "
            "if a grid or scoring change improved this, tighten the xfail "
            "below instead of loosening this bound"
        )

    # The ≥90 % off-node recovery SPEC that lived here as a strict xfail
    # flipped on 2026-07-02 (stack-rework Phase 2): it now PASSES against
    # the coarse-to-fine blind search — see
    # test_coarse_fine.py::TestOffNodeSpecFlipped.  The classical Stacker
    # keeps the coarse-grid limitation by design (auditable reference);
    # the characterization tests above pin that behaviour.


class TestCriterionControl:
    """Positive control: the from_prior sampling criterion is sufficient."""

    def test_criterion_sampled_grid_recovers_off_node_target(self) -> None:
        """A grid at step ≈ PSF_fwhm/T recovers the worst-case target.

        This is the quantitative basis for the Phase 2 coarse-to-fine
        design: correctness needs criterion sampling, and criterion
        sampling of the FULL ±400 px/s box is what is intractable (F2) —
        not criterion sampling per se.
        """
        frames_off = _gaussian_frames(_V_OFF_NODE, 0.0)
        fine = Stacker.from_prior(
            vx0_px_s=_V_OFF_NODE + _FINE_STEP / 3.0,  # deliberately off-node
            vy0_px_s=0.0,
            half_width_px_s=2.0,
            step_px_s=_FINE_STEP,
        )
        result = fine.stack(frames_off, _NOISE_RMS, _times_s())
        assert result.stacked.max() > 0.90 * _N_FRAMES * _AMPLITUDE
        assert result.best_vx_px_s == pytest.approx(_V_OFF_NODE, abs=_FINE_STEP)

    def test_smear_matches_dwell_time_model(self) -> None:
        """The off-node peak loss follows the dwell-time prediction.

        For smear path L ≫ PSF_fwhm the stacked profile approaches a
        uniform trail: peak ≈ N·A·√(2π)·σ/L.  With L = 50 px and
        σ = 1.2 px that is ≈ 6 % of N·A.  Verify within a factor ≈ 2
        (discreteness of 25 samples along the trail).
        """
        frames_off = _gaussian_frames(_V_OFF_NODE, 0.0)
        result = _coarse_stacker().stack(frames_off, _NOISE_RMS, _times_s())
        smear_len_px = (_COARSE_STEP / 2.0) * _T_SPAN_S  # 50 px
        predicted = _N_FRAMES * _AMPLITUDE * math.sqrt(2 * math.pi) * (
            _PSF_SIGMA_PX / smear_len_px
        )
        measured = result.stacked.max()
        assert 0.5 * predicted < measured < 2.0 * predicted, (
            f"measured {measured:.0f} vs dwell-time prediction {predicted:.0f}"
        )
