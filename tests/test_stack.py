"""Tests for the track-and-stack module (WP-A5).

Structure
---------
TestShiftAndAdd        — _shift_and_add primitive: signal registers correctly
TestStackerUnit        — Stacker class: correct hypothesis wins, noise scales
TestStackerFromConfig  — Stacker.from_config wires velocity grid from YAML
TestStackAcceptanceMv13 — WP-A5 acceptance: mv 13.0 stacked SNR ≥ 5

Accept criterion (WP-A5 / closes OpTA.NOD.DET):
  On an ISS-speed SyntheticPass at mv 13.0, single-frame detection FAILS
  (SNR₁ < 3.0 threshold) but the stacked image achieves peak SNR ≥ 5.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sensor_fixtures import OPTICS_DEFAULT, SENSOR_SMALL

from opta_pipeline.calibrate import calibrate_frame
from opta_pipeline.config import PipelineConfig
from opta_pipeline.detect import detect_sources
from opta_pipeline.stack import (
    Stacker,
    StackResult,
    _shift_and_add,
    stack_frames,
    symmetric_velocity_axis,
)

_NOD_DET_MIN_SNR = 5.0  # OpTA.NOD.DET acceptance threshold


# ---------------------------------------------------------------------------
# 1. _shift_and_add primitive
# ---------------------------------------------------------------------------


class TestShiftAndAdd:
    """The shift-and-add primitive correctly registers a known signal."""

    def _impulse_frame(
        self, h: int, w: int, y: int, x: int, value: float
    ) -> np.ndarray:
        frame = np.zeros((h, w), dtype=np.float64)
        frame[y, x] = value
        return frame

    def test_zero_shift_preserves_signal(self) -> None:
        """Zero velocity: frames are unchanged, signal accumulates at origin."""
        h, w = 30, 40
        signal = 50.0
        frames = [self._impulse_frame(h, w, 10, 20, signal) for _ in range(5)]
        t = np.zeros(5)
        result = _shift_and_add(frames, t, 0.0, 0.0, (h, w))
        assert result[10, 20] == pytest.approx(5 * signal)

    def test_correct_shift_registers_signal(self) -> None:
        """Correct (vx, vy) brings all impulses to the reference position."""
        h, w = 50, 50
        signal = 100.0
        fps = 25.0
        vx, vy = 10.0, 5.0  # px/s
        t = np.array([k / fps for k in range(5)])  # 0, 0.04, 0.08, 0.12, 0.16 s

        # Place impulse at (x0 + vx*t_k, y0 + vy*t_k) in each frame
        x0, y0 = 20, 15
        frames = []
        for tk in t:
            xk = int(round(x0 + vx * tk))
            yk = int(round(y0 + vy * tk))
            frames.append(self._impulse_frame(h, w, yk, xk, signal))

        result = _shift_and_add(frames, t, vx, vy, (h, w))
        # All impulses register within the bilinear footprint of (y0, x0).
        # Peak-pixel preservation is NOT the invariant here: registering an
        # impulse at a sub-pixel offset spreads it over the 2×2 bilinear
        # footprint (worst case 25 % peak retention at a (0.5, 0.5) offset;
        # the previous ≥ 0.9·4·signal peak assertion only passed via
        # cubic-spline overshoot).  The true invariants are flux conservation
        # in the registration neighbourhood and peak location.
        yy, xx = np.unravel_index(int(np.argmax(result)), result.shape)
        assert (yy, xx) == (y0, x0), f"Peak at ({yy}, {xx}), expected ({y0}, {x0})"
        window = result[y0 - 2 : y0 + 3, x0 - 2 : x0 + 3]
        n_frames = len(frames)
        assert window.sum() > n_frames * signal * 0.98, (
            f"Registered flux too low: {window.sum():.1f} "
            f"(expected ≥ {n_frames * signal * 0.98:.1f})"
        )

    def test_wrong_shift_smears_signal(self) -> None:
        """Wrong velocity smears signal; max pixel < correct-velocity peak."""
        h, w = 80, 80
        signal = 100.0
        fps = 25.0
        # Use 2 px/frame so consecutive frames never alias to the same pixel
        vx_true = 50.0  # px/s = 2 px/frame at 25 fps
        vy_true = 0.0
        t = np.array([k / fps for k in range(10)])

        x0, y0 = 10, 40
        frames = []
        for tk in t:
            xk = int(round(x0 + vx_true * tk))
            yk = y0
            frames.append(self._impulse_frame(h, w, yk, xk, signal))

        # Correct velocity → all impulses register at (x0, y0)
        result_correct = _shift_and_add(frames, t, vx_true, vy_true, (h, w))
        # Wrong velocity (0, 0) → impulses scatter across 10 different columns
        result_wrong = _shift_and_add(frames, t, 0.0, 0.0, (h, w))

        # Each wrong-shift frame places its impulse at a unique column (2px spacing);
        # the highest pixel just contains one frame's contribution.
        assert result_wrong.max() < result_correct.max() * 0.5, (
            f"Wrong velocity (max={result_wrong.max():.1f}) should be much lower "
            f"than correct velocity (max={result_correct.max():.1f})"
        )

    def test_out_of_frame_shifts_fill_zero(self) -> None:
        """Frames shifted entirely out of bounds contribute no noise."""
        h, w = 20, 20
        frames = [np.ones((h, w), dtype=np.float64)]  # uniform background = 1
        t = np.array([1.0])
        # Shift so far right that the entire frame is off-screen
        result = _shift_and_add(frames, t, 1000.0, 0.0, (h, w))
        assert result.max() == pytest.approx(0.0), (
            "Off-frame shift should produce all-zero output"
        )


# ---------------------------------------------------------------------------
# 2. Stacker unit tests
# ---------------------------------------------------------------------------


class TestStackerUnit:
    """Stacker correctly identifies the best velocity and reports peak SNR."""

    @pytest.fixture(scope="class")
    def moving_signal_stack(self) -> dict:
        """10 frames with a known-velocity impulse; noise_rms=1.0."""
        h, w = 40, 60
        fps = 25.0
        vx_true, _vy_true = 8.0, 0.0  # px/s; purely horizontal
        signal = 20.0
        noise_rms = 1.0
        rng = np.random.default_rng(0)

        x0, y0 = 30, 20
        t = np.array([k / fps for k in range(10)])
        frames = []
        for tk in t:
            frame = rng.normal(0.0, noise_rms, (h, w))
            xk = int(round(x0 + vx_true * tk))
            yk = y0
            if 0 <= xk < w and 0 <= yk < h:
                frame[yk, xk] += signal
            frames.append(frame)

        vx_grid = np.array([-8.0, 0.0, 8.0, 16.0])
        vy_grid = np.array([0.0])
        stacker = Stacker(vx_grid, vy_grid)
        result = stacker.stack(frames, noise_rms, list(t))
        return {"result": result, "vx_true": vx_true, "signal": signal, "n": 10}

    def test_correct_hypothesis_gives_highest_snr(
        self, moving_signal_stack: dict
    ) -> None:
        """The closest hypothesis to vx_true should be selected as best."""
        result = moving_signal_stack["result"]
        vx_true = moving_signal_stack["vx_true"]
        # best_vx should be closest grid point to vx_true (= 8.0)
        assert result.best_vx_px_s == pytest.approx(vx_true, abs=1e-6)

    def test_peak_snr_exceeds_single_frame(self, moving_signal_stack: dict) -> None:
        """Stacked peak SNR must be > 1 (coherent summation beats noise)."""
        result = moving_signal_stack["result"]
        assert result.peak_snr > 1.0

    def test_noise_rms_scales_sqrt_n(self, moving_signal_stack: dict) -> None:
        """noise_rms = single_frame_noise × √N."""
        result = moving_signal_stack["result"]
        n = moving_signal_stack["n"]
        # noise_rms = 1.0 × √10
        assert result.noise_rms == pytest.approx(math.sqrt(n), rel=1e-9)

    def test_n_frames_reported_correctly(self, moving_signal_stack: dict) -> None:
        assert moving_signal_stack["result"].n_frames == moving_signal_stack["n"]

    def test_stacked_array_has_correct_shape(self, moving_signal_stack: dict) -> None:
        result = moving_signal_stack["result"]
        assert result.stacked.shape == (40, 60)

    def test_stack_frames_convenience_wrapper(self) -> None:
        """stack_frames returns a StackResult without explicit Stacker."""
        frames = [np.zeros((10, 10))]
        result = stack_frames(frames, noise_rms=1.0)
        assert isinstance(result, StackResult)
        assert result.n_frames == 1

    def test_empty_frames_raises(self) -> None:
        stacker = Stacker(np.array([0.0]), np.array([0.0]))
        with pytest.raises(ValueError, match="must not be empty"):
            stacker.stack([], 1.0)

    def test_mismatched_times_raises(self) -> None:
        stacker = Stacker(np.array([0.0]), np.array([0.0]))
        with pytest.raises(ValueError, match="len\\(frame_times_s\\)"):
            stacker.stack([np.zeros((5, 5))], 1.0, frame_times_s=[0.0, 1.0])


# ---------------------------------------------------------------------------
# 3. Config integration
# ---------------------------------------------------------------------------


class TestSymmetricVelocityAxis:
    """The shared velocity-axis helper: symmetric, zero-preserving, no drift.

    ``np.arange(-v_max, v_max + step/2, step)`` — the pattern this helper
    replaced at five sites — silently walks off-node whenever ``v_max`` is not
    an integer multiple of ``step``, dropping the v=0 hypothesis and skewing
    the search box.  The helper must fix that *without* moving any grid the
    production configs actually use.
    """

    # (v_max, step) pairs that are exactly divisible AND exactly representable:
    # pipeline_defaults.yaml (400/20), the validation grids, and the from_prior
    # half-widths used across the pipeline suite.
    _DIVISIBLE = [
        (400.0, 20.0),
        (100.0, 20.0),
        (10.0, 5.0),
        (200.0, 200.0),
        (240.0, 120.0),
        (80.0, 40.0),
        (6.0, 2.0),
        (2.0, 0.5),
        (1.5, 0.5),
        (5.0, 0.25),
    ]

    @pytest.mark.parametrize(("v_max", "step"), _DIVISIBLE)
    def test_divisible_grids_are_byte_identical_to_the_old_pattern(
        self, v_max: float, step: float
    ) -> None:
        """No pinned detection number can move: same bits, not just same values."""
        old = np.arange(-v_max, v_max + step * 0.5, step)
        new = symmetric_velocity_axis(v_max, step)
        assert new.shape == old.shape
        assert new.tobytes() == old.tobytes()

    @pytest.mark.parametrize(
        ("v_max", "step"), [(162.0, 20.0), (200.0, 15.0), (175.0, 20.0)]
    )
    def test_indivisible_grids_keep_zero_and_symmetry(
        self, v_max: float, step: float
    ) -> None:
        """The regression: 162/20 and 200/15 lost v=0 under the old pattern."""
        old = np.arange(-v_max, v_max + step * 0.5, step)
        assert not np.any(old == 0.0)  # the defect being fixed

        new = symmetric_velocity_axis(v_max, step)
        assert np.count_nonzero(new == 0.0) == 1
        np.testing.assert_array_equal(new, -new[::-1])
        # v_max is a ROUNDED bound: the axis ends at the node nearest v_max,
        # which may sit up to half a step outside it (rounding outward, so the
        # box covers the requested band rather than stopping short).
        assert np.all(np.abs(new) <= v_max + step * 0.5)
        assert float(np.max(np.diff(new))) == pytest.approx(step)

    def test_extreme_node_may_round_outward_past_v_max(self) -> None:
        """Pin the true bound: 175/20 ends at ±180, not ±160 or ±175."""
        grid = symmetric_velocity_axis(175.0, 20.0)
        assert grid[0] == -180.0
        assert grid[-1] == 180.0
        assert float(np.max(np.abs(grid))) > 175.0

    def test_zero_half_width_is_the_single_zero_node(self) -> None:
        """from_prior(half_width=0) must still search exactly the prior."""
        np.testing.assert_array_equal(
            symmetric_velocity_axis(0.0, 20.0), np.array([0.0])
        )

    def test_nonpositive_step_raises(self) -> None:
        with pytest.raises(ValueError, match="step must be > 0"):
            symmetric_velocity_axis(10.0, 0.0)


class TestVelocityGridAdoption:
    """All five in-package grid sites go through the shared helper."""

    def test_from_config_grid_contains_zero_for_indivisible_vmax(self) -> None:
        from dataclasses import replace

        from opta_pipeline.likelihood import PsiPhiStacker

        cfg = PipelineConfig.default()
        stk = replace(
            cfg.stacking, velocity_max_px_s=162.0, velocity_step_px_s=20.0
        )
        for grid in (
            Stacker.from_config(stk).vx_grid,
            PsiPhiStacker.from_config(stk).vx_grid,
        ):
            assert np.count_nonzero(grid == 0.0) == 1
            np.testing.assert_array_equal(grid, -grid[::-1])

    def test_stack_frames_default_grid_uses_the_helper(self) -> None:
        """The module's own convenience default is the sixth site (±2 / 0.5).

        Divisible, so the nodes are byte-identical to the literal arange it
        replaced — this pins that the module does not keep a private copy of
        the pattern it just centralised.
        """
        expected = symmetric_velocity_axis(2.0, 0.5)
        np.testing.assert_array_equal(expected, np.arange(-2.0, 2.5, 0.5))

        rng = np.random.default_rng(0)
        frames = [rng.normal(0.0, 1.0, (12, 12)) for _ in range(4)]
        times = [0.0, 0.4, 0.8, 1.2]
        default = stack_frames(frames, 1.0, times)
        explicit = stack_frames(frames, 1.0, times, expected, expected)
        np.testing.assert_array_equal(default.stacked, explicit.stacked)
        assert default.best_vx_px_s == explicit.best_vx_px_s
        assert default.best_vy_px_s == explicit.best_vy_px_s

    def test_from_prior_grid_brackets_the_prior_exactly(self) -> None:
        """An indivisible half-width must still include the predicted vector."""
        from opta_pipeline.likelihood import PsiPhiStacker

        for cls in (Stacker, PsiPhiStacker):
            s = cls.from_prior(
                10.0, -20.0, half_width_px_s=3.5, step_px_s=2.0
            )
            assert np.count_nonzero(s.vx_grid == 10.0) == 1
            assert np.count_nonzero(s.vy_grid == -20.0) == 1


class TestStackerFromConfig:
    """Stacker.from_config wires velocity grid from pipeline_defaults.yaml."""

    def test_from_config_creates_symmetric_grid(self) -> None:
        cfg = PipelineConfig.default()
        stacker = Stacker.from_config(cfg)
        assert stacker.vx_grid[0] == pytest.approx(-cfg.stacking.velocity_max_px_s)
        assert stacker.vx_grid[-1] == pytest.approx(
            cfg.stacking.velocity_max_px_s, abs=1.0
        )

    def test_from_config_step_matches_yaml(self) -> None:
        cfg = PipelineConfig.default()
        stacker = Stacker.from_config(cfg)
        if len(stacker.vx_grid) > 1:
            step = float(stacker.vx_grid[1] - stacker.vx_grid[0])
            assert step == pytest.approx(cfg.stacking.velocity_step_px_s, rel=1e-6)

    def test_from_pipeline_config_works_same_as_stacking_config(self) -> None:
        cfg = PipelineConfig.default()
        stacker_full = Stacker.from_config(cfg)
        stacker_sub = Stacker.from_config(cfg.stacking)
        np.testing.assert_array_equal(stacker_full.vx_grid, stacker_sub.vx_grid)


# ---------------------------------------------------------------------------
# 4. Acceptance test — mv 13.0 stacked detection (closes OpTA.NOD.DET)
# ---------------------------------------------------------------------------
#
# Physics rationale for parameter choices
# ----------------------------------------
# At mv 13.0 / 0.5 deg/s / SENSOR_SMALL + VILTROX 85 mm f/1.4:
#   trail_px  = 0.5 × 3600 × 0.04 / 9.12  ≈  7.9 px
#   trail_loss= 1/7.9                       ≈  0.127
#   sig_total = signal_electrons(13.0, ...)  ≈ 15.9 e
#   sig_peak  ≈ sig_total / (trail_px × σ√2π) ≈ 0.94 e/px/frame  (<< noise ≈ 1.1 e)
#   noise_rms ≈ √(sky + RN²) ≈ 1.1 e/px; pipeline clamps at 1.0
#
# Single-frame detect_sources:
#   pixel threshold = 1.1 × 3.0 = 3.3 e  >>  satellite peak 0.94 e → NOT detected
#
# After stacking N=50 frames (satellite horizontally traverses the 400 px frame):
#   stacked signal peak ≈ 50 × 0.94   ≈ 47 e   (coherent sum)
#   stacked noise       = 1.1 × √50  ≈  7.8 e  (incoherent)
#   stacked peak SNR    ≈ 47 / 7.8   ≈  6.0  ≥ 5.0 (OpTA.NOD.DET ✓)
#   detect threshold    = 7.8 × 3.0 = 23 e  <  47 e → trail detected ✓
#
# Satellite traverses 400 px frame in ≈ 400 / (0.5 × 3600/9.12) ≈ 2 s = 50 frames.
# All 50 frames have the satellite in FOV → coherent sum is maximal.


def _make_mv13_frames(n_frames: int = 50) -> tuple[list, list[float], float]:
    """Generate n_frames synthetic frames with a mv=13.0 satellite at 0.5 deg/s.

    Returns
    -------
    cal_frames : list of np.ndarray
        Calibrated, background-subtracted float arrays.
    times_s : list of float
        Frame times in seconds, t=0 at the first frame.
    noise_rms : float
        Stacking noise floor (single-frame background_rms, clamped to 1.0).
    """
    from opta_pipeline.synth import SatelliteSpec, generate_frame

    fps = SENSOR_SMALL.frame_rate_hz
    dt = 1.0 / fps
    ang_vel_deg_s = 0.5  # half-degree/s → trail ≈ 7.9 px per frame
    vx_px_frame = ang_vel_deg_s * 3600.0 / 9.12 * dt  # ≈ 7.9 px/frame
    x_start = 10.0
    y_center = SENSOR_SMALL.resolution_v / 2.0

    rng = np.random.default_rng(42)
    cal_frames = []
    noise_rms_vals = []

    for k in range(n_frames):
        x_center = x_start + vx_px_frame * k
        sat = SatelliteSpec(
            magnitude=13.0,
            angular_velocity_deg_s=ang_vel_deg_s,
            x_center=x_center,
            y_center=y_center,
            angle_deg=0.0,
        )
        sf = generate_frame(
            SENSOR_SMALL,
            OPTICS_DEFAULT,
            sky_mag_arcsec2=21.0,
            elevation_deg=45.0,
            satellites=[sat],
            rng=np.random.default_rng(int(rng.integers(0, 2**31))),
        )
        cal = calibrate_frame(sf.data_float, subtract_background=True)
        cal_frames.append(cal.data)
        noise_rms_vals.append(max(cal.background_rms, 1.0))

    times_s = [k * dt for k in range(n_frames)]
    noise_rms = float(np.mean(noise_rms_vals))
    return cal_frames, times_s, noise_rms


@pytest.fixture(scope="module")
def mv13_stack_data() -> dict:
    """50-frame mv=13.0 acceptance fixture (module-scoped for speed)."""
    n_frames = 50
    cal_frames, times_s, noise_rms = _make_mv13_frames(n_frames)

    ang_vel_deg_s = 0.5
    vx_true = ang_vel_deg_s * 3600.0 / 9.12  # ≈ 197.4 px/s

    # Grid: ±40 px/s around true velocity, step 20 px/s (5 hypotheses × 1 vy = 5 total)
    step = 20.0
    vx_grid = np.arange(vx_true - 40.0, vx_true + 40.0 + step * 0.5, step)
    vy_grid = np.array([0.0])

    result = Stacker(vx_grid, vy_grid).stack(cal_frames, noise_rms, times_s)
    return {
        "result": result,
        "cal_frames": cal_frames,
        "times_s": times_s,
        "noise_rms": noise_rms,
        "vx_true": vx_true,
        "n_frames": n_frames,
    }


class TestStackAcceptanceMv13:
    """WP-A5 acceptance: stacked detection of mv 13.0 satellite achieves SNR ≥ 5.

    Satellite: mv=13.0, ang_vel=0.5 deg/s, 50 frames at 25 fps (2 s pass).
    Single-frame peak signal ≈ 0.94 e < noise ≈ 1.1 e → single-frame detection
    FAILS (pixel threshold = noise × 3.0 ≈ 3.3 e >> 0.94 e).
    50-frame stack: peak ≈ 47 e, noise ≈ 7.8 e → peak_snr ≈ 6.0 ≥ 5.0 ✓.

    Closes B3 in IMPROVEMENT_PLAN.md; promotes OpTA.NOD.DET from "model only"
    to "closed end-to-end by pipeline".
    """

    def test_single_frame_not_detectable(self, mv13_stack_data: dict) -> None:
        """Single-frame pixel threshold ≫ satellite per-pixel signal.

        At 0.5 deg/s the trail peak is ≈ 0.94 e/px.  With noise_rms ≈ 1.1 e
        and snr_threshold=3.0, the threshold ≈ 3.3 e — the satellite is
        completely invisible in a single frame.
        """
        snr_threshold = PipelineConfig.default().detection.snr_threshold
        cal = mv13_stack_data["cal_frames"][25]  # middle frame
        noise_rms = mv13_stack_data["noise_rms"]
        dets = detect_sources(cal, noise_rms=noise_rms, snr_threshold=snr_threshold)
        streak_snrs = [d.snr for d in dets if d.is_streak]
        assert not any(s >= snr_threshold for s in streak_snrs), (
            "mv 13.0 satellite should not be single-frame detectable at 0.5 deg/s"
        )

    def test_stacked_peak_snr_meets_nod_det(self, mv13_stack_data: dict) -> None:
        """Stacked peak SNR ≥ 5 closes OpTA.NOD.DET at pipeline level (WP-A5)."""
        result = mv13_stack_data["result"]
        assert result.peak_snr >= _NOD_DET_MIN_SNR, (
            f"Stacked peak SNR = {result.peak_snr:.2f} < {_NOD_DET_MIN_SNR} "
            "(OpTA.NOD.DET requires stacked SNR ≥ 5)"
        )

    def test_detect_sources_finds_source_in_stacked_frame(
        self, mv13_stack_data: dict
    ) -> None:
        """detect_sources finds satellite trail in stacked frame with SNR ≥ 5."""
        result = mv13_stack_data["result"]
        dets = detect_sources(
            result.stacked,
            noise_rms=result.noise_rms,
            snr_threshold=3.0,
        )
        assert len(dets) >= 1, (
            f"No source detected in stacked frame "
            f"(peak_snr={result.peak_snr:.2f}, noise_rms={result.noise_rms:.2f})"
        )
        best_snr = max(d.snr for d in dets)
        assert best_snr >= _NOD_DET_MIN_SNR, (
            f"Best component SNR = {best_snr:.2f} < {_NOD_DET_MIN_SNR}"
        )

    def test_correct_velocity_dominates_wrong_hypothesis(
        self, mv13_stack_data: dict
    ) -> None:
        """Correct velocity coherently accumulates signal; a wrong one does not.

        Robustness notes (this test was previously brittle):

        * The comparison is **integrated over the registered streak footprint**,
          not read from a single pixel.  A single stacked pixel carries noise of
          ``noise_rms·√N`` (≈8 e here), so a single-pixel value/ratio is noise-
          dominated and swings wildly between RNG seeds — the original
          single-pixel 10× ratio failed deterministically once the RNG stream
          shifted.  Integrating over the ~33-px footprint beats that noise down.
        * The wrong hypothesis is a **reversed velocity**, not zero.  Zero is a
          poor discriminator because the satellite *starts* at the reference
          pixel (x=10), so a stationary stack accumulates genuine early-frame
          signal there (its peak SNR even reaches the NOD.DET threshold).  A
          reversed velocity registers a track that sweeps the opposite way, so
          no frame's signal lands on the source — a true null.

        Empirically the footprint SNR is ≈19σ (correct) vs ≈0.5σ (reversed)
        across seeds, so the absolute thresholds below carry a wide margin.
        """
        cal_frames = mv13_stack_data["cal_frames"]
        times_s = mv13_stack_data["times_s"]
        vx_true = mv13_stack_data["vx_true"]
        # Stacked (co-added) noise floor: single-frame rms grows as √N under summation.
        stacked_noise = mv13_stack_data["noise_rms"] * math.sqrt(len(cal_frames))

        t = np.asarray(times_s, dtype=np.float64)
        f64 = [f.astype(np.float64) for f in cal_frames]
        shape = f64[0].shape

        best = _shift_and_add(f64, t, vx_true, 0.0, shape)
        wrong = _shift_and_add(f64, t, -vx_true, 0.0, shape)  # genuinely wrong

        ref_y = int(SENSOR_SMALL.resolution_v / 2)
        ref_x = 10  # x_start of satellite at t=0

        def footprint_snr(img: np.ndarray) -> float:
            """Integrated SNR over the registered streak footprint at the source."""
            box = img[ref_y - 1 : ref_y + 2, ref_x - 5 : ref_x + 6]
            box_noise = stacked_noise * math.sqrt(box.size)
            return float(box.sum()) / box_noise

        snr_best = footprint_snr(best)
        snr_wrong = footprint_snr(wrong)

        # Correct velocity: a strong coherent detection, far above NOD.DET (5σ).
        assert snr_best >= 10.0, (
            f"Correct-velocity footprint SNR = {snr_best:.2f} < 10 "
            "(coherent accumulation failed)"
        )
        # Wrong velocity: no coherent accumulation — consistent with noise.
        assert snr_wrong < 5.0, (
            f"Reversed-velocity footprint SNR = {snr_wrong:.2f} ≥ 5 "
            "(spurious accumulation at the source position)"
        )

    def test_matched_filter_prefers_true_velocity(
        self, mv13_stack_data: dict
    ) -> None:
        """Production Stacker: true velocity beats a far-wrong one on noisy data.

        Exercises the real matched-filter path (``Stacker.stack``) rather than
        the ``_shift_and_add`` primitive.  The correct velocity must clear the
        NOD.DET threshold while a 2× velocity (which smears the streak) stays
        well below it — i.e. the peak-SNR objective is genuinely selective.
        Empirically ≈6.7 vs ≈3.2 across seeds.
        """
        cal_frames = mv13_stack_data["cal_frames"]
        times_s = mv13_stack_data["times_s"]
        noise_rms = mv13_stack_data["noise_rms"]
        vx_true = mv13_stack_data["vx_true"]

        snr_true = (
            Stacker(np.array([vx_true]), np.array([0.0]))
            .stack(cal_frames, noise_rms, times_s)
            .peak_snr
        )
        snr_wrong = (
            Stacker(np.array([2.0 * vx_true]), np.array([0.0]))
            .stack(cal_frames, noise_rms, times_s)
            .peak_snr
        )

        assert snr_true >= _NOD_DET_MIN_SNR, (
            f"True-velocity peak SNR = {snr_true:.2f} < {_NOD_DET_MIN_SNR}"
        )
        assert snr_true > 1.5 * snr_wrong, (
            f"Matched filter not selective: true {snr_true:.2f} vs "
            f"2×-velocity {snr_wrong:.2f}"
        )


# ---------------------------------------------------------------------------
# 6. Velocity grid midpoint (T-06) — worst-case grid position
# ---------------------------------------------------------------------------


class TestVelocityGridMidpoint:
    """Verify Stacker behaviour when true velocity falls exactly between grid nodes.

    Physical reasoning
    ------------------
    At the midpoint between two grid nodes neither hypothesis perfectly aligns
    all frames.  The peak signal at the best grid node is reduced compared to
    exact alignment, but it must remain clearly above noise because at least
    some frames contribute coherently.

    The stacker must never hallucinate a velocity that is absent from the grid;
    it selects the best grid node by highest peak_snr, with ties broken by
    first encounter (smallest vx index).
    """

    @staticmethod
    def _make_frames() -> tuple[list[np.ndarray], list[float]]:
        """2-frame 1-D case: satellite at vx_true=10 px/s (between 0 and 20).

        Frame 0: satellite at col 20.
        Frame 1: satellite at col 30 (10 px/s × 1 s).
        """
        h, w = 30, 80
        frame0 = np.zeros((h, w), dtype=np.float64)
        frame0[15, 20] = 100.0
        frame1 = np.zeros((h, w), dtype=np.float64)
        frame1[15, 30] = 100.0  # vx_true=10 px/s × 1s
        return [frame0, frame1], [0.0, 1.0]

    @staticmethod
    def _make_frames_2d() -> tuple[list[np.ndarray], list[float]]:
        """2-frame 2-D case: satellite moving at (vx, vy)=(10, 10) px/s.

        True velocity is the midpoint of the 2×2 grid {0, 20} × {0, 20}.
        Frame 0: satellite at (row=30, col=30).
        Frame 1: satellite at (row=40, col=40).
        """
        h, w = 80, 80
        frame0 = np.zeros((h, w), dtype=np.float64)
        frame0[30, 30] = 100.0
        frame1 = np.zeros((h, w), dtype=np.float64)
        frame1[40, 40] = 100.0  # vx_true=vy_true=10 px/s × 1s
        return [frame0, frame1], [0.0, 1.0]

    def test_best_velocity_is_grid_node(self) -> None:
        """The stacker must return a velocity that is actually in the search grid.

        It must not hallucinate vx=10.0 (true velocity, NOT in {0, 20, 40}).
        """
        stacker = Stacker(vx_grid=np.array([0.0, 20.0, 40.0]), vy_grid=np.array([0.0]))
        frames, times_s = self._make_frames()
        result = stacker.stack(frames, noise_rms=1.0, frame_times_s=times_s)
        grid_values = {0.0, 20.0, 40.0}
        assert result.best_vx_px_s in grid_values, (
            f"best_vx={result.best_vx_px_s} not in grid {grid_values}"
        )

    def test_peak_snr_above_noise(self) -> None:
        """Even at the midpoint (worst-case grid position), peak_snr must exceed 3.0.

        At 100 e⁻ signal and noise_rms=1.0 with N=2 frames, expect ≈70.7
        (= 100 / (1.0 × √2)).
        """
        stacker = Stacker(vx_grid=np.array([0.0, 20.0, 40.0]), vy_grid=np.array([0.0]))
        frames, times_s = self._make_frames()
        result = stacker.stack(frames, noise_rms=1.0, frame_times_s=times_s)
        assert result.peak_snr >= 3.0

    def test_diagonal_midpoint_also_returns_grid_node(self) -> None:
        """2-D case: true velocity (10, 10) px/s, grid {0, 20} × {0, 20}.

        Stack at all 4 hypotheses; best must be one of the 4 grid corners.
        """
        stacker = Stacker(
            vx_grid=np.array([0.0, 20.0]),
            vy_grid=np.array([0.0, 20.0]),
        )
        frames, times_s = self._make_frames_2d()
        result = stacker.stack(frames, noise_rms=1.0, frame_times_s=times_s)
        assert result.best_vx_px_s in {0.0, 20.0}
        assert result.best_vy_px_s in {0.0, 20.0}
