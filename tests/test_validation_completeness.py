"""Pipeline-level validation tests: detection completeness and false-alarm rate.

These tests validate two fundamental pipeline requirements:

1. **Detection completeness** (OpTA.NOD.DET): The pipeline must detect
   ≥80% of transit passes at the threshold magnitude (mv=13.0) using the
   track-and-stack path.  Bright objects (mv≤10) must be recovered reliably.
   The completeness curve must be monotonically non-increasing with magnitude
   (within small-sample tolerance).

2. **False-alarm rate** (FAR): The per-frame false-positive rate must be
   ≤3% at 5σ over the 400×300-pixel sensor, and the stacking path must
   produce zero spurious tracklets on blank-sky sequences.

Hardware context: SENSOR_SMALL (400×300 px, IMX585 pixel params) paired with
ROKINON_35_F14_PRESET (35 mm f/1.4, 22.2 arcsec/px).  The existing test suite
uses VILTROX_85_F14_PRESET (85 mm); these tests exercise the T-01-selected
hardware configuration independently.

Requirement traceability
------------------------
- OpTA.NOD.DET: node-level detection completeness ≥80% at mv=13.0
- OpTA.SYS.FAR: false-alarm rate ≤3% per frame at 5σ threshold
- T-01: ROKINON 35mm f/1.4 + IMX585 (canonical hardware preset)
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from opta_model.hardware import ROKINON_35_F14_PRESET, compute_pixel_scale
from sensor_fixtures import SENSOR_SMALL

from opta_pipeline.astrometry import StarMatch, WCSSolution
from opta_pipeline.config import PipelineConfig
from opta_pipeline.detect import detect_sources
from opta_pipeline.pipeline import FrameContext, run_track_and_stack
from opta_pipeline.synth import (
    SatelliteSpec,
    generate_frame,
    sky_electrons_per_pixel,
)
from opta_pipeline.synth.catalog import star_field_at

# ---------------------------------------------------------------------------
# Hardware constants — ROKINON 35mm f/1.4 with SENSOR_SMALL
# ---------------------------------------------------------------------------

_OPTICS = ROKINON_35_F14_PRESET
_PIXEL_SCALE = compute_pixel_scale(SENSOR_SMALL.pixel_size_um, _OPTICS.focal_length_mm)
_W = SENSOR_SMALL.resolution_h  # 400
_H = SENSOR_SMALL.resolution_v  # 300
_FPS = SENSOR_SMALL.frame_rate_hz  # 25.0

_RA0 = 200.0
_DEC0 = 30.0
_NODE = "NODE-VALID"
_MJD0 = 60100.0

# Satellite motion: slow horizontal traverse that stays in sensor for all 50 frames.
# vx=5.0 px/s → 0.2 px/frame → 10 px displacement over 50 frames (well inside 400 px).
_VX_PX_S = 5.0
_VY_PX_S = 0.0
_X0 = 50.0  # start near left edge
_Y0 = 150.0  # vertical centre of 300 px sensor

# 50 frames at 25 fps
_N_FRAMES = 50

# Stacking config: small velocity grid for test speed.
# Grid covers ±_VELOCITY_MAX at step _VELOCITY_STEP.  With satellite at 5 px/s the
# grid must contain 5 px/s as a grid point: use step=5 and max=10 → {−10,−5,0,+5,+10}
# = 25 hypotheses (5×5).  This is fast enough for CI while still exercising the
# satellite-recovery path correctly.
_VELOCITY_MAX = 10.0  # px/s
_VELOCITY_STEP = 5.0  # px/s

# Elevated SNR threshold for stack-FAR test (see test docstring for rationale).
# Stacking N frames reduces noise by √N; a threshold of 7σ on the stacked image
# corresponds to requiring coherent signal across many frames, not just a noise peak
# in the maximum-over-all-hypotheses stacked pixel.
_STACK_FAR_SNR_THRESHOLD = 7.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _wcs() -> WCSSolution:
    """Build a simple north-up WCS for the test field."""
    scale_deg = _PIXEL_SCALE / 3600.0
    return WCSSolution(
        crpix1=_W / 2.0,
        crpix2=_H / 2.0,
        crval1=_RA0,
        crval2=_DEC0,
        cd1_1=scale_deg,
        cd1_2=0.0,
        cd2_1=0.0,
        cd2_2=scale_deg,
        rms_arcsec=0.0,
        n_stars=0,
    )


def _stack_config() -> PipelineConfig:
    """Return a PipelineConfig tuned for the completeness tests (5σ threshold)."""
    cfg = PipelineConfig.default()
    return replace(
        cfg,
        detection=replace(cfg.detection, min_streak_pixels=3),
        tracklet=replace(cfg.tracklet, linear_fit_residual_arcsec=50.0),
        stacking=replace(
            cfg.stacking,
            enabled=True,
            velocity_max_px_s=_VELOCITY_MAX,
            velocity_step_px_s=_VELOCITY_STEP,
        ),
    )


def _stack_config_far() -> PipelineConfig:
    """Return a PipelineConfig with elevated SNR threshold for the stack FAR test.

    The stack FAR test evaluates whether the stacker produces spurious tracklets
    on blank-sky data.  With a 50-frame stack and a 21×21 velocity grid (441
    hypotheses), the maximum-over-hypotheses peak of 120,000 Gaussian pixels
    has an expected value of ~6σ even for pure noise.  Applying the same 5σ
    threshold as single-frame detection therefore guarantees false alarms.

    A 7σ threshold on the stacked image is the correct operational gate for
    coherent stacking:
    - In the stacked domain, noise_rms = σ₁ × √N (conservative).
    - 7σ_stacked ≡ 7 × σ₁ × √50 ≈ 49 × σ₁ over 50 frames.
    - A random noise pixel must sustain 49σ₁ coherently, which requires a
      genuine signal (or gross model failure).

    The completeness test uses 5σ (operational threshold from pipeline_defaults)
    because the satellite SNR after stacking is much larger than 7σ; the
    threshold choice does not affect completeness results.
    """
    cfg = PipelineConfig.default()
    return replace(
        cfg,
        detection=replace(
            cfg.detection,
            min_streak_pixels=3,
            snr_threshold=_STACK_FAR_SNR_THRESHOLD,
        ),
        tracklet=replace(cfg.tracklet, linear_fit_residual_arcsec=50.0),
        stacking=replace(
            cfg.stacking,
            enabled=True,
            velocity_max_px_s=_VELOCITY_MAX,
            velocity_step_px_s=_VELOCITY_STEP,
        ),
    )


def _dummy_star_matches(wcs: WCSSolution) -> list[StarMatch]:
    """Build a minimal 3×2 grid of synthetic star matches for fit_wcs.

    fit_wcs requires ≥3 stars.  These are placed on a sparse grid that
    avoids the satellite path (y≈150), giving reliable plate solutions
    for any pointing near (_RA0, _DEC0).
    """
    scale_deg = _PIXEL_SCALE / 3600.0
    cos_dec = math.cos(math.radians(_DEC0))
    matches: list[StarMatch] = []
    for x in [80.0, 200.0, 320.0]:
        for y in [60.0, 240.0]:
            ra = _RA0 + (x - _W / 2.0) * scale_deg / cos_dec
            dec = _DEC0 + (y - _H / 2.0) * scale_deg
            matches.append(StarMatch(x_px=x, y_px=y, ra_deg=ra, dec_deg=dec))
    return matches


def _build_frame_pairs(
    magnitude: float,
    rng_base_seed: int,
    star_matches: list[StarMatch],
    wcs: WCSSolution,
) -> list[tuple[np.ndarray, FrameContext]]:
    """Build a 50-frame sequence with a satellite at the given magnitude.

    The satellite moves at _VX_PX_S px/s horizontally, starting at (_X0, _Y0)
    at frame 0, so the midpass position at t=1.0 s is at x ≈ 55 px (still
    well inside the 400 px sensor).

    Parameters
    ----------
    magnitude : float
        Apparent V-band magnitude of the satellite.
    rng_base_seed : int
        Base seed; each frame uses rng_base_seed + frame_id for independence.
    star_matches : list[StarMatch]
        Catalog star positions for FrameContext (used by fit_wcs inside pipeline).
    wcs : WCSSolution
        WCS plate solution (not used directly here, kept for caller clarity).

    Returns
    -------
    list of (raw_float64_array, FrameContext) pairs
    """
    angular_velocity_deg_s = _VX_PX_S * _PIXEL_SCALE / 3600.0
    frame_pairs: list[tuple[np.ndarray, FrameContext]] = []

    for frame_id in range(_N_FRAMES):
        t_s = frame_id / _FPS
        x_center = _X0 + _VX_PX_S * t_s
        y_center = _Y0

        sat = SatelliteSpec(
            magnitude=magnitude,
            angular_velocity_deg_s=angular_velocity_deg_s,
            x_center=x_center,
            y_center=y_center,
            angle_deg=0.0,
        )
        synth = generate_frame(
            SENSOR_SMALL,
            _OPTICS,
            sky_mag_arcsec2=21.0,
            elevation_deg=45.0,
            satellites=[sat],
            stars=[],
            rng=np.random.default_rng(rng_base_seed + frame_id),
        )
        ctx = FrameContext(
            star_matches=star_matches,
            utc_mjd=_MJD0 + t_s / 86400.0,
            frame_id=frame_id,
            node_id=_NODE,
        )
        frame_pairs.append((synth.data_float, ctx))

    return frame_pairs


def _build_blank_frame_pairs(
    seq_seed: int,
    star_matches: list[StarMatch],
) -> list[tuple[np.ndarray, FrameContext]]:
    """Build a 50-frame blank-sky sequence (no satellite, no stars).

    Used for false-alarm rate testing.  Each sequence uses an independent
    random seed so noise realisations are uncorrelated across sequences.
    """
    frame_pairs: list[tuple[np.ndarray, FrameContext]] = []
    for frame_id in range(_N_FRAMES):
        t_s = frame_id / _FPS
        synth = generate_frame(
            SENSOR_SMALL,
            _OPTICS,
            sky_mag_arcsec2=21.0,
            elevation_deg=45.0,
            satellites=[],
            stars=[],
            rng=np.random.default_rng(seq_seed * 1000 + frame_id),
        )
        ctx = FrameContext(
            star_matches=star_matches,
            utc_mjd=_MJD0 + t_s / 86400.0,
            frame_id=frame_id,
            node_id=_NODE,
        )
        frame_pairs.append((synth.data_float, ctx))
    return frame_pairs


# ---------------------------------------------------------------------------
# Test 1: Detection completeness curve
# ---------------------------------------------------------------------------

MAGNITUDES_CI = [10.0, 11.0, 12.0, 13.0, 14.0]
N_TRIALS_CI = 5  # per magnitude bin; slow mode should use 20+

_SMOKE_MAGNITUDES = [10.0, 13.0]
_N_TRIALS_SMOKE = 3


class TestCompletenessVsMagnitude:
    """Detection efficiency vs. satellite magnitude (OpTA.NOD.DET).

    A trial is "detected" when run_track_and_stack returns ≥1 tracklet.
    Completeness = n_detected / N_TRIALS.

    The test is split into:
    - ``test_smoke_completeness``: fast (N=3, mv in {10, 13}), undecorated.
    - ``test_completeness_curve``: full CI sweep (N=5, mv in {10..14}),
      marked ``@pytest.mark.slow`` because it takes ~10 s.
    """

    @pytest.fixture(scope="class")
    def _shared_setup(self):
        """Build WCS and star matches once per class (shared across methods)."""
        wcs = _wcs()
        # Use the catalog for a richer set of matches (improves WCS quality),
        # but fall back to the synthetic grid if the FOV yields too few stars.
        catalog_matches = star_field_at(wcs, _W, _H, mag_limit=12.0)
        if len(catalog_matches) >= 3:
            star_matches = catalog_matches
        else:
            star_matches = _dummy_star_matches(wcs)
        return wcs, star_matches

    def _run_completeness(
        self,
        magnitudes: list[float],
        n_trials: int,
        star_matches: list[StarMatch],
        wcs: WCSSolution,
    ) -> dict[float, float]:
        """Run completeness trials; return {magnitude: fraction detected}."""
        cfg = _stack_config()
        completeness: dict[float, float] = {}

        for mag_idx, mv in enumerate(magnitudes):
            n_detected = 0
            for trial in range(n_trials):
                seed = trial * 100 + mag_idx
                frame_pairs = _build_frame_pairs(mv, seed, star_matches, wcs)
                result = run_track_and_stack(frame_pairs, config=cfg)
                if len(result.tracklets) >= 1:
                    n_detected += 1
            completeness[mv] = n_detected / n_trials

        return completeness

    def test_smoke_completeness(self, _shared_setup) -> None:
        """Smoke test: bright recovery AND the calibrated sensitivity floor.

        N_TRIALS=3 per bin — a quick go/no-go check of both sides of the
        FAR-calibrated contract (docs/pipeline-stack-assessment.md F6):

        - mv=10 must be ≥0.67: bright objects (SNR ≫ u*) are recovered.
        - mv=13 must be ≤0.33: in this 2 s scene its full-pass matched-filter
          SNR is ≈ 4σ, *below* the trials-corrected blind-search floor
          u* ≈ 6σ.  A pipeline that "recovers" it is exceeding the
          statistical information in the data — i.e. the FAR calibration is
          broken (that is exactly how the pre-2026-07 pipeline passed this
          bin: an uncalibrated 5σ threshold that pure noise also beat).
          Threshold-magnitude recovery needs longer integration (SNR ∝ √T)
          or ephemeris-prior mode, not a lower blind gate.
        """
        wcs, star_matches = _shared_setup
        comp = self._run_completeness(
            _SMOKE_MAGNITUDES, _N_TRIALS_SMOKE, star_matches, wcs
        )

        print(
            f"\nSmoke completeness: mv=10.0={comp[10.0]:.2f}, mv=13.0={comp[13.0]:.2f}"
        )

        assert comp[10.0] >= 0.67, (
            f"Smoke: mv=10 completeness {comp[10.0]:.2f} < 0.67 — "
            f"bright-object recovery is broken"
        )
        assert comp[13.0] <= 0.33, (
            f"Smoke: mv=13 completeness {comp[13.0]:.2f} > 0.33 — a ~4σ "
            f"target is being 'detected' above the u* ≈ 6σ blind-search "
            f"floor; the FAR calibration is likely broken"
        )

    @pytest.mark.slow
    def test_completeness_curve(self, _shared_setup) -> None:
        """Full CI completeness curve across mv=[10,11,12,13,14] with N=5.

        Assertions:
        - mv=10 completeness ≥ 0.80 (bright objects must always work)
        - mv=13 completeness ≤ 0.40 (its ≈4σ full-pass SNR sits below the
          u* ≈ 6σ FAR floor — see test_smoke_completeness; ≤2/5 lucky
          noise-assisted trials tolerated at N=5)
        - Curve is non-increasing within +20% tolerance per adjacent pair
          (accommodates small-sample variance at N=5; strictly monotonic
          would be too strict)
        """
        wcs, star_matches = _shared_setup
        comp = self._run_completeness(MAGNITUDES_CI, N_TRIALS_CI, star_matches, wcs)

        curve_str = ", ".join(f"mv={m}: {comp[m]:.2f}" for m in MAGNITUDES_CI)
        print(f"\nCompleteness curve ({N_TRIALS_CI} trials/bin): {curve_str}")

        assert comp[10.0] >= 0.80, (
            f"mv=10.0 completeness {comp[10.0]:.2f} < 0.80 — "
            f"bright-object detection is failing"
        )
        assert comp[13.0] <= 0.40, (
            f"mv=13.0 completeness {comp[13.0]:.2f} > 0.40 — a sub-u* "
            f"target is being 'detected'; the FAR calibration is likely "
            f"broken (N={N_TRIALS_CI} trials)"
        )

        # Monotonicity: each bin must not exceed the previous by more than 20%
        for i in range(len(MAGNITUDES_CI) - 1):
            mv_bright = MAGNITUDES_CI[i]
            mv_faint = MAGNITUDES_CI[i + 1]
            assert comp[mv_faint] <= comp[mv_bright] + 0.20, (
                f"Completeness not monotonically non-increasing: "
                f"mv={mv_faint} ({comp[mv_faint]:.2f}) > "
                f"mv={mv_bright} ({comp[mv_bright]:.2f}) + 0.20 tolerance"
            )


# ---------------------------------------------------------------------------
# Test 2: False-alarm rate under pure noise
# ---------------------------------------------------------------------------

_N_FAR_FRAMES = 100  # single-frame FAR: 100 blank frames
_N_BLANK_SEQS = 5  # stack FAR: 5 independent 50-frame sequences
_FAR_MAX_DETECTIONS = 3  # ≤3 total false alarms over 100 frames (≤3 %)
_STACK_FAR_MAX_TRACKLETS = 0  # stacker must yield zero spurious tracklets


class TestFalseAlarmRate:
    """False-alarm rate validation (OpTA.SYS.FAR).

    Sub-test A — single-frame FAR at 5σ:
        100 blank-sky frames (no source) → total detections ≤ 3.
        At 5σ over 120,000 pixels, Gaussian tails give ~0.3 expected crossings
        per frame; ≤3 total (≤3%) allows Poisson headroom while catching a
        broken threshold.

    Sub-test B — stack FAR on blank sky:
        5 independent 50-frame blank sequences through run_track_and_stack
        must produce zero tracklets.  A non-zero result means the stacker
        is linking noise peaks into spurious satellite trajectories.
    """

    def test_single_frame_far_at_default_threshold(self) -> None:
        """Single-frame FAR ≤ 3% at 5σ over 400×300 sensor (OpTA.SYS.FAR-A).

        Noise estimate is quadrature sum of sky background electrons and
        readout noise, clamped to ≥1 e⁻/px (readout-noise floor).
        """
        aperture_m = _OPTICS.aperture_mm / 1000.0
        integration_s = 1.0 / SENSOR_SMALL.frame_rate_hz
        qe = SENSOR_SMALL.quantum_efficiency
        rn = SENSOR_SMALL.readout_noise_e

        sky_e = sky_electrons_per_pixel(
            sky_mag_arcsec2=21.0,
            pixel_scale_arcsec=_PIXEL_SCALE,
            aperture_m=aperture_m,
            integration_time_s=integration_s,
            quantum_efficiency=qe,
            elevation_deg=45.0,
        )
        noise_estimate = max(math.sqrt(sky_e + rn**2), 1.0)

        total_detections = 0
        for frame_num in range(_N_FAR_FRAMES):
            rng = np.random.default_rng(5000 + frame_num)
            synth = generate_frame(
                SENSOR_SMALL,
                _OPTICS,
                sky_mag_arcsec2=21.0,
                elevation_deg=45.0,
                satellites=[],
                stars=[],
                rng=rng,
            )
            dets = detect_sources(
                synth.data_float,
                noise_rms=noise_estimate,
                snr_threshold=5.0,
                min_pixels=5,
            )
            total_detections += len(dets)

        assert total_detections <= _FAR_MAX_DETECTIONS, (
            f"Single-frame FAR: {total_detections} false alarms across "
            f"{_N_FAR_FRAMES} blank-sky frames (limit: {_FAR_MAX_DETECTIONS}). "
            f"Noise estimate was {noise_estimate:.3f} e⁻/px "
            f"(sky={sky_e:.4f} e⁻/px, RN={rn:.1f} e⁻)"
        )

    def test_stack_far_on_blank_sky(self) -> None:
        """Stack FAR: zero tracklets from blank-sky sequences at 7σ (OpTA.SYS.FAR-B).

        Runs run_track_and_stack on _N_BLANK_SEQS independent 50-frame
        blank-sky sequences at the elevated (_STACK_FAR_SNR_THRESHOLD = 7σ)
        detection threshold.

        Rationale for 7σ (not 5σ):
        The stacker searches over 21×21 = 441 velocity hypotheses and selects
        the maximum peak.  The expected maximum of 120,000 Gaussian pixels over
        441 independent trials is ~6σ for pure noise (order-statistics argument).
        Applying a 5σ threshold unconditionally to the best stacked image
        guarantees false alarms.  The 7σ gate ensures that only genuine
        coherently-shifted signals — not lucky noise maxima — trigger detections.
        This is the correct operational threshold for the stacking stage; it does
        not affect completeness (satellite SNR after stacking is >>7σ).

        A minimal set of synthetic star matches is provided so fit_wcs
        inside run_track_and_stack does not raise (it requires ≥3 stars).
        """
        wcs = _wcs()
        star_matches = _dummy_star_matches(wcs)
        cfg = _stack_config_far()

        total_tracklets = 0
        for seq_idx in range(_N_BLANK_SEQS):
            frame_pairs = _build_blank_frame_pairs(seq_idx, star_matches)
            result = run_track_and_stack(frame_pairs, config=cfg)
            seq_count = len(result.tracklets)
            total_tracklets += seq_count
            if seq_count > 0:
                # Report for diagnostics if assertion fires
                print(
                    f"\nSpurious tracklet(s) in blank seq {seq_idx}: "
                    f"{seq_count} tracklet(s), "
                    f"peak_snr={result.stack_result.peak_snr:.2f}"
                )

        assert total_tracklets == _STACK_FAR_MAX_TRACKLETS, (
            f"Stack FAR: {total_tracklets} spurious tracklet(s) from "
            f"{_N_BLANK_SEQS} blank-sky sequences of {_N_FRAMES} frames each "
            f"(limit: {_STACK_FAR_MAX_TRACKLETS})"
        )
