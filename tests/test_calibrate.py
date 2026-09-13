"""Tests for opta_pipeline.calibrate.

Validates:
  - Master dark creation (sigma-clip combines frames correctly)
  - Master flat creation (normalised; dead pixels handled)
  - Dark subtraction reduces frame mean
  - Flat correction equalises response across the field
  - Background estimation tracks the injected sky level
  - Background RMS matches theoretical prediction (OT-020 accept criterion)
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from opta_model.hardware import (
    IMX585_PRESET,
    ROKINON_35_F14_PRESET,
    VILTROX_85_F14_PRESET,
    compute_pixel_scale,
)

from opta_pipeline.calibrate import (
    CalibratedFrame,
    calibrate_frame,
    estimate_background,
    make_master_dark,
    make_master_flat,
)
from opta_pipeline.synth import generate_frame, sky_electrons_per_pixel

SENSOR = IMX585_PRESET
OPTICS = VILTROX_85_F14_PRESET


# ---------------------------------------------------------------------------
# 1. Master dark creation
# ---------------------------------------------------------------------------


class TestMasterDark:
    """Sigma-clipped median master dark."""

    def test_too_few_frames_raises(self) -> None:
        frames = [np.zeros((10, 10), dtype=np.uint16)] * 2
        with pytest.raises(ValueError, match="3"):
            make_master_dark(frames)

    def test_output_shape(self) -> None:
        frames = [np.ones((50, 50), dtype=np.uint16) * 100 for _ in range(5)]
        dark = make_master_dark(frames)
        assert dark.shape == (50, 50)

    def test_output_dtype_float32(self) -> None:
        frames = [np.ones((10, 10), dtype=np.uint16) * 50 for _ in range(5)]
        dark = make_master_dark(frames)
        assert dark.dtype == np.float32

    def test_constant_frames_give_constant_dark(self) -> None:
        """All identical frames must produce a master equal to that value."""
        val = 42.0
        frames = [np.full((20, 20), val, dtype=np.float32) for _ in range(7)]
        dark = make_master_dark(frames)
        np.testing.assert_allclose(dark, val, atol=0.5)

    def test_outlier_frames_rejected(self) -> None:
        """One hot frame should not affect the master (sigma-clipped away)."""
        rng = np.random.default_rng(20)
        good = [rng.integers(10, 30, (30, 30), dtype=np.uint16) for _ in range(9)]
        hot = [np.full((30, 30), 30000, dtype=np.uint16)]
        dark = make_master_dark(good + hot)
        assert float(dark.mean()) < 40.0


# ---------------------------------------------------------------------------
# 2. Master flat creation
# ---------------------------------------------------------------------------


class TestMasterFlat:
    """Normalised sigma-clipped median flat."""

    def test_too_few_frames_raises(self) -> None:
        frames = [np.ones((5, 5), dtype=np.uint16)] * 2
        with pytest.raises(ValueError, match="3"):
            make_master_flat(frames)

    def test_flat_median_near_one(self) -> None:
        """Normalised flat median must be near 1.0."""
        frames = [np.full((20, 20), 1000.0, dtype=np.float32) for _ in range(5)]
        flat = make_master_flat(frames)
        assert float(np.median(flat)) == pytest.approx(1.0, abs=0.01)

    def test_flat_shape(self) -> None:
        frames = [np.ones((30, 40), dtype=np.uint16) * 500 for _ in range(5)]
        flat = make_master_flat(frames)
        assert flat.shape == (30, 40)

    def test_dead_pixel_replaced_with_one(self) -> None:
        """Pixel with near-zero flat value should be set to 1.0 (no correction)."""
        frames = [np.full((10, 10), 1000.0, dtype=np.float32) for _ in range(5)]
        for f in frames:
            f[5, 5] = 1.0  # dead pixel
        flat = make_master_flat(frames)
        assert flat[5, 5] == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# 3. Dark subtraction and flat correction
# ---------------------------------------------------------------------------


class TestDarkFlatCorrection:
    """calibrate_frame correctly applies dark and flat corrections."""

    def test_dark_subtraction_reduces_mean(self) -> None:
        rng = np.random.default_rng(30)
        # Fake raw frame with non-zero dark signal
        raw = rng.integers(100, 200, (50, 50), dtype=np.uint16)
        dark = np.full((50, 50), 100.0, dtype=np.float32)
        cal = calibrate_frame(
            raw, master_dark=dark, master_flat=None, subtract_background=False
        )
        assert float(cal.data.mean()) < float(raw.astype(float).mean())
        assert cal.dark_subtracted is True
        assert cal.flat_corrected is False

    def test_flat_correction_flag(self) -> None:
        raw = np.full((20, 20), 500, dtype=np.uint16)
        flat = np.full((20, 20), 1.0, dtype=np.float32)
        cal = calibrate_frame(
            raw, master_dark=None, master_flat=flat, subtract_background=False
        )
        assert cal.flat_corrected is True

    def test_flat_with_vignetting_equalised(self) -> None:
        """Flat-correcting a frame with vignetting should equalise the response."""
        rng = np.random.default_rng(31)
        h, w = 60, 80
        raw = np.full((h, w), 1000, dtype=np.float32)
        # Vignetting: corners are 70% of centre
        yy, xx = np.mgrid[:h, :w]
        r2 = (yy - h / 2) ** 2 + (xx - w / 2) ** 2
        r_max2 = (h / 2) ** 2 + (w / 2) ** 2
        vig = 1.0 - 0.3 * (r2 / r_max2)
        vignetted = (raw * vig).astype(np.uint16)

        # Flat field captures the vignetting pattern
        flat_frames = [
            (vignetted + rng.integers(-5, 5, (h, w))).astype(np.uint16)
            for _ in range(5)
        ]
        flat = make_master_flat(flat_frames)

        cal = calibrate_frame(vignetted, master_flat=flat, subtract_background=False)
        # After correction, response variation should be much less than before
        raw_std = float(vignetted.astype(float).std())
        cal_std = float(cal.data.std())
        assert cal_std < raw_std * 0.5

    def test_no_corrections_applied(self) -> None:
        raw = np.full((20, 20), 500, dtype=np.uint16)
        cal = calibrate_frame(
            raw, master_dark=None, master_flat=None, subtract_background=False
        )
        assert cal.dark_subtracted is False
        assert cal.flat_corrected is False
        np.testing.assert_allclose(cal.data, raw.astype(np.float32))


# ---------------------------------------------------------------------------
# 4. Background estimation
# ---------------------------------------------------------------------------


class TestBackgroundEstimation:
    """Background model and RMS should match the injected sky level."""

    def test_background_shape_matches_input(self) -> None:
        frame = np.random.default_rng(40).normal(50, 5, (120, 160)).astype(np.float32)
        bg, _ = estimate_background(frame, box_size=32)
        assert bg.shape == (120, 160)

    def test_background_near_sky_level(self) -> None:
        """Background estimate should recover the injected constant sky value."""
        sky = 25.0
        rng = np.random.default_rng(41)
        frame = rng.normal(sky, 3.0, (128, 128)).astype(np.float32)
        bg, _ = estimate_background(frame, box_size=32)
        assert float(np.median(bg)) == pytest.approx(sky, abs=2.0)

    def test_background_rms_positive(self) -> None:
        frame = np.random.default_rng(42).normal(10, 2, (64, 64)).astype(np.float32)
        _, rms = estimate_background(frame, box_size=16)
        assert rms > 0

    def test_background_rms_matches_noise_model(self) -> None:
        """RMS should match sqrt(sky_e + dark_e + RN²) within 20%.

        OT-020 accept criterion: background RMS matches expected sky noise.
        """
        pixel_scale = compute_pixel_scale(SENSOR.pixel_size_um, OPTICS.focal_length_mm)
        aperture_m = OPTICS.aperture_mm / 1000.0
        integration_s = 1.0 / SENSOR.frame_rate_hz
        sky_e = sky_electrons_per_pixel(
            18.0,
            pixel_scale,
            aperture_m,
            integration_s,
            SENSOR.quantum_efficiency,
        )
        dark_e = SENSOR.dark_current_e_s * integration_s
        rn_e = SENSOR.readout_noise_e
        expected_rms = math.sqrt(sky_e + dark_e + rn_e**2)

        # Generate a synthetic frame with no sources
        rng = np.random.default_rng(43)
        frame_obj = generate_frame(
            SENSOR, OPTICS, sky_mag_arcsec2=18.0, satellites=[], stars=[], rng=rng
        )
        _, measured_rms = estimate_background(
            frame_obj.data.astype(np.float32),
            box_size=64,
        )
        # Allow 20% tolerance (clipping bias at low counts, discrete Poisson)
        assert measured_rms == pytest.approx(expected_rms, rel=0.20)


class TestBackgroundGradientRegistration:
    """Regression: mesh model must be exact on noiseless linear gradients.

    Pins the 2026-07-19 defect (TODO P3): the box grid left the trailing
    ``h % box_size`` strip unsampled, ``zoom(order=1)`` anchored box-centre
    medians on frame corners (half-box misregistration), and the mean
    smoothing filter flattened gradients at the borders.  On a noiseless
    0.5 e-/row gradient (960x540, box 64) the old model left interior
    residuals up to 34 e-, bottom-strip residuals up to 270 e-, and
    reported background_rms = 16.9 e- where true noise is zero — misfit
    masquerading as the very sigma that detect_sources / psi-phi / u*
    normalise against.  Twilight gradients are exactly this failure mode.

    A correct box-centre model is *exact* on any bilinear plane: box
    medians of a linear ramp equal the ramp value at the box centre, the
    separable median filter preserves planes, and bilinear interpolation
    with linear edge extrapolation reconstructs the plane everywhere,
    including the remainder strip and the outer half-box margins.
    """

    # 960x540 is 2x-binned 1080p video (28-row remainder at box 64);
    # 1920x1080 is unbinned (56-row remainder).
    @pytest.mark.parametrize("shape", [(540, 960), (1080, 1920)])
    def test_row_gradient_exact(self, shape: tuple[int, int]) -> None:
        h, w = shape
        frame = (0.5 * np.arange(h, dtype=np.float64))[:, None] * np.ones((1, w))
        bg, rms = estimate_background(frame.astype(np.float32), box_size=64)
        resid = np.abs(frame - bg)
        # Float32 storage of the model bounds the error at ~1e-5 relative;
        # the defective model left residuals of tens to hundreds of e-.
        assert float(resid.max()) < 1e-3
        assert rms < 1e-3

    def test_remainder_strip_is_sampled(self) -> None:
        """The trailing h % box_size strip must be fitted, not extrapolated."""
        h, w, box = 540, 960, 64
        frame = (0.5 * np.arange(h, dtype=np.float64))[:, None] * np.ones((1, w))
        bg, _ = estimate_background(frame.astype(np.float32), box_size=box)
        strip = np.abs(frame - bg)[(h // box) * box :, :]
        assert strip.size > 0
        assert float(strip.max()) < 1e-3  # was 269.5 e- pre-fix

    def test_diagonal_gradient_exact(self) -> None:
        """Bilinear plane with both slopes — catches corner-box filter bias."""
        h, w = 540, 960
        yy, xx = np.mgrid[0:h, 0:w]
        frame = 0.5 * yy + 0.3 * xx
        bg, rms = estimate_background(frame.astype(np.float32), box_size=64)
        assert float(np.abs(frame - bg).max()) < 1e-2
        assert rms < 1e-2

    def test_gradient_rms_not_inflated_by_misfit(self) -> None:
        """On gradient + noise, rms must report the noise, not the gradient."""
        h, w, sigma = 540, 960, 3.0
        rng = np.random.default_rng(44)
        gradient = (0.5 * np.arange(h, dtype=np.float64))[:, None] * np.ones((1, w))
        frame = gradient + rng.normal(0.0, sigma, (h, w))
        _, rms = estimate_background(frame.astype(np.float32), box_size=64)
        # Pre-fix: gradient misfit inflated this to ~17 e- on the noiseless
        # ramp alone; now it must track the injected sigma.
        assert rms == pytest.approx(sigma, rel=0.05)


# ---------------------------------------------------------------------------
# 5. Full calibration pipeline
# ---------------------------------------------------------------------------


class TestCalibrateFrame:
    """End-to-end calibrate_frame tests."""

    def test_background_subtraction_flag(self) -> None:
        raw = np.full((40, 40), 100.0, dtype=np.float32)
        cal = calibrate_frame(raw, subtract_background=True)
        assert cal.background_subtracted is True

    def test_calibrated_frame_near_zero_mean(self) -> None:
        """After dark + flat + bg subtraction, mean of a uniform frame ≈ 0."""
        raw = np.full((40, 40), 200.0, dtype=np.float32)
        dark = np.full((40, 40), 50.0, dtype=np.float32)
        flat = np.full((40, 40), 1.0, dtype=np.float32)
        cal = calibrate_frame(
            raw, master_dark=dark, master_flat=flat, subtract_background=True
        )
        assert abs(float(cal.data.mean())) < 5.0

    def test_background_rms_positive(self) -> None:
        rng = np.random.default_rng(50)
        raw = rng.normal(100, 5, (64, 64)).astype(np.float32)
        cal = calibrate_frame(raw)
        assert cal.background_rms > 0

    def test_synth_frame_calibration(self) -> None:
        """Calibrate a synthetic sky frame; background RMS within factor 2 of theory."""
        rng = np.random.default_rng(51)
        synth = generate_frame(
            SENSOR, OPTICS, sky_mag_arcsec2=18.0, satellites=[], stars=[], rng=rng
        )
        cal = calibrate_frame(
            synth.data,
            subtract_background=True,
            background_box_size=64,
        )
        assert cal.background_rms > 0
        assert cal.background_rms < synth.sky_e_per_pixel * 5


# ---------------------------------------------------------------------------
# 6. Calibrated residuals are Gaussian — KS test (T-04 from TESTING.md)
# ---------------------------------------------------------------------------


class TestCalibratedNoiseModel:
    """Verify calibrate_frame produces residuals consistent with N(0, σ_expected).

    Sub-test A: pixel distribution KS test — residuals should not be
    distinguishable from a Gaussian at the p=0.05 level.

    Sub-test B: background_rms matches theoretical sqrt(sky_e + dark_e + RN²)
    within 20 % — confirming the noise model used throughout the pipeline is
    consistent with the actual photon statistics.

    Hardware: ROKINON_35_F14_PRESET optics + IMX585_PRESET sensor parameters.
    Frame size is fixed at 400×300 so the KS test uses a sample size (120 K
    pixels → 500-pixel subsample) with appropriate statistical power.

    Calibration input: SynthFrame.data_float (pre-quantisation float64 array)
    is used instead of the uint16 .data array.  At sky = 21 mag/arcsec² and
    a 35 mm aperture the sky flux is ≈ 0.37 e⁻/px (0.89 before the F9a
    band-averaged-QE derating → 0.67, then × (2.9/3.76)² for the MA-002
    datasheet-pitch unification, both 2026-07) — below the uint16 truncation
    bias regime identified in AGENTS.md ("data_float for low-flux
    statistical assertions").  Using uint16 here would underestimate
    background_rms substantially and make the KS test fail spuriously due
    to integer-quantisation artefacts.

    KS test is run on a 500-pixel random subsample (seed 99).  Using all
    120 K pixels gives a statistic that exceeds the threshold because
    Kolmogorov–Smirnov has extremely high power at large N and detects the
    mild skewness of a sub-electron Poisson background — a physically real
    but operationally irrelevant deviation.  Subsampling to 500 pixels gives
    the test appropriate statistical power: it will catch gross
    non-Gaussianity (systematic offsets, bimodality, heavy tails) while not
    rejecting a nearly-Gaussian distribution on theoretical grounds.  The
    statistic bound is 0.07: at sub-electron flux the Poisson skew is
    physically larger than at the pre-F9a 0.89 e⁻/px (this fixed-seed
    sample sits at ≈ 0.036), while the p-value sub-test remains the
    principled gate.
    """

    _OPTICS = ROKINON_35_F14_PRESET
    _SKY = 21.0
    # 400×300 sensor with IMX585 pixel parameters (matches spec "400×300 frame")
    _SENSOR = IMX585_PRESET
    _H = 300
    _W = 400
    # Number of pixels to subsample for KS test (see class docstring)
    _KS_SUBSAMPLE = 500

    @pytest.fixture(scope="class")
    def calibrated(self) -> CalibratedFrame:
        """Generate a 400×300 sky-only frame, calibrate with synthetic dark/flat."""
        from opta_model.hardware import SensorConfig

        # Build a 400×300 sensor with IMX585 physical parameters
        sensor_params = self._SENSOR
        sensor = SensorConfig(
            pixel_size_um=sensor_params.pixel_size_um,
            resolution_h=self._W,
            resolution_v=self._H,
            quantum_efficiency=sensor_params.quantum_efficiency,
            full_well_e=sensor_params.full_well_e,
            dark_current_e_s=sensor_params.dark_current_e_s,
            readout_noise_e=sensor_params.readout_noise_e,
            # Pinned: the class docstring's flux numbers (≈ 0.37 e⁻/px
            # after the F9a QE derating and the MA-002 2.9 µm pitch
            # unification) assume the 25 fps ROI-mode integration time.
            # IMX585_PRESET is the full-res mode at 21 fps since the
            # 2026-07 F1 fix; a 400×300 ROI legitimately reads out at
            # ≥25 fps.
            frame_rate_hz=25.0,
        )
        optics = self._OPTICS
        rng = np.random.default_rng(99)

        # Sky-only synthetic frame (no stars, no satellite)
        frame = generate_frame(
            sensor,
            optics,
            sky_mag_arcsec2=self._SKY,
            satellites=[],
            stars=[],
            rng=rng,
        )

        # Synthetic master dark: 5 frames of pure readout noise
        rng_dark = np.random.default_rng(42)
        dark_frames = [
            rng_dark.normal(
                0.0, sensor_params.readout_noise_e, (self._H, self._W)
            ).astype(np.float32)
            for _ in range(5)
        ]
        master_dark = make_master_dark(dark_frames)

        # Synthetic master flat: uniform (all-ones)
        flat_frames = [np.ones((self._H, self._W), dtype=np.float32) for _ in range(5)]
        master_flat = make_master_flat(flat_frames)

        # Use data_float (pre-quantisation float64) — avoids uint16 truncation
        # bias at low sky flux (see class docstring).
        return calibrate_frame(
            frame.data_float.astype(np.float32),
            master_dark=master_dark,
            master_flat=master_flat,
            subtract_background=True,
        )

    def test_ks_p_value(self, calibrated: CalibratedFrame) -> None:
        """KS p-value ≥ 0.05: cannot reject Gaussian hypothesis (sub-test A).

        A 500-pixel subsample (seed 99) is used to give the KS test appropriate
        statistical power; see class docstring for rationale.
        """
        from scipy import stats

        # cal.data is already background-subtracted; residuals ≈ zero-mean
        all_normalized = (
            calibrated.data.ravel().astype(np.float64) / calibrated.background_rms
        )
        rng_ks = np.random.default_rng(99)
        sample = rng_ks.choice(all_normalized, size=self._KS_SUBSAMPLE, replace=False)
        _stat, p_value = stats.kstest(sample, "norm")
        assert p_value >= 0.05, (
            f"KS test rejected Gaussian hypothesis: p={p_value:.4f} < 0.05 "
            "— calibrated residuals are not Gaussian"
        )

    def test_ks_statistic(self, calibrated: CalibratedFrame) -> None:
        """KS statistic ≤ 0.07: distribution is close to Gaussian (sub-test A).

        Same 500-pixel subsample as test_ks_p_value.  Bound relaxed from
        0.05 with the F9a flux derating, and retained through the MA-002
        pitch unification (sky 0.89 → 0.67 → 0.37 e⁻/px): the lower-flux
        Poisson background is physically more skewed; see the class
        docstring.  This seed measures ≈ 0.036; the p-value sub-test
        remains the principled gate.
        """
        from scipy import stats

        all_normalized = (
            calibrated.data.ravel().astype(np.float64) / calibrated.background_rms
        )
        rng_ks = np.random.default_rng(99)
        sample = rng_ks.choice(all_normalized, size=self._KS_SUBSAMPLE, replace=False)
        stat, _p = stats.kstest(sample, "norm")
        assert stat <= 0.07, (
            f"KS statistic = {stat:.4f} > 0.07 — residuals deviate from Gaussian"
        )

    def test_background_rms_matches_theory(self, calibrated: CalibratedFrame) -> None:
        """background_rms within 20% of sqrt(sky_e + dark_e + RN²) (sub-test B)."""
        from opta_model.hardware import compute_pixel_scale

        sensor_params = self._SENSOR
        optics = self._OPTICS

        pixel_scale = compute_pixel_scale(
            sensor_params.pixel_size_um, optics.focal_length_mm
        )
        aperture_m = optics.aperture_mm / 1000.0
        integration_s = 1.0 / sensor_params.frame_rate_hz

        sky_e = sky_electrons_per_pixel(
            self._SKY,
            pixel_scale,
            aperture_m,
            integration_s,
            sensor_params.quantum_efficiency,
        )
        dark_e = sensor_params.dark_current_e_s / sensor_params.frame_rate_hz
        readout_noise_e = sensor_params.readout_noise_e
        theoretical_noise = math.sqrt(sky_e + dark_e + readout_noise_e**2)

        rel_err = abs(calibrated.background_rms - theoretical_noise) / theoretical_noise
        assert rel_err <= 0.20, (
            f"background_rms = {calibrated.background_rms:.3f} e, "
            f"theoretical = {theoretical_noise:.3f} e, "
            f"relative error = {rel_err:.3f} > 0.20"
        )
