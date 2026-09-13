"""Tests for opta_pipeline.synth — synthetic frame generator.

Validates that the synth module:
  - Produces frames with correct shape and dtype
  - Noise statistics match the expected sky/dark/readout model
  - Injected stellar sources have the correct integrated signal
  - Injected satellite streaks carry the radiometrically-correct per-pixel signal
  - FITS round-trip preserves data exactly
  - FITS header contains all pipeline_defaults.yaml required fields
  - SNR at the detection limit is near threshold (single frame)

Ground truth: uses IMX585_PRESET + VILTROX_85_F14_PRESET (T-01/T-02 selected hardware).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from opta_model.hardware import (
    IMX585_PRESET,
    VILTROX_85_F14_PRESET,
    compute_pixel_scale,
)
from opta_model.radiometry import compute_snr

from opta_pipeline.synth import (
    SatelliteSpec,
    StarSpec,
    generate_frame,
    satellite_signal_electrons,
    sky_electrons_per_pixel,
    write_fits,
)

# Fixed RNG seed for reproducibility
_RNG = np.random.default_rng(42)

# Selected hardware (T-01 + T-02)
SENSOR = IMX585_PRESET
OPTICS = VILTROX_85_F14_PRESET


# ---------------------------------------------------------------------------
# 1. Frame shape and dtype
# ---------------------------------------------------------------------------


class TestFrameShape:
    """Output frame must match the sensor resolution and be uint16."""

    def test_frame_shape(self) -> None:
        frame = generate_frame(SENSOR, OPTICS, rng=np.random.default_rng(0))
        assert frame.data.shape == (SENSOR.resolution_v, SENSOR.resolution_h)

    def test_frame_dtype(self) -> None:
        frame = generate_frame(SENSOR, OPTICS, rng=np.random.default_rng(0))
        assert frame.data.dtype == np.uint16

    def test_pixel_scale_matches_hardware(self) -> None:
        """Stored pixel scale must match the hardware.py formula."""
        expected = compute_pixel_scale(SENSOR.pixel_size_um, OPTICS.focal_length_mm)
        frame = generate_frame(SENSOR, OPTICS, rng=np.random.default_rng(0))
        assert frame.pixel_scale_arcsec == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 2. Noise statistics
# ---------------------------------------------------------------------------


class TestNoiseStatistics:
    """Background statistics should reflect the radiometric noise model."""

    def test_sky_electrons_positive(self) -> None:
        frame = generate_frame(
            SENSOR, OPTICS, sky_mag_arcsec2=21.0, rng=np.random.default_rng(1)
        )
        assert frame.sky_e_per_pixel > 0

    def test_darker_sky_fewer_background_electrons(self) -> None:
        """Bortle 5 (20.5) should have more sky background than Bortle 3 (21.5)."""
        sky_bright = sky_electrons_per_pixel(
            20.5,
            frame_e := compute_pixel_scale(
                SENSOR.pixel_size_um, OPTICS.focal_length_mm
            ),
            OPTICS.aperture_mm / 1000.0,
            1.0 / SENSOR.frame_rate_hz,
            SENSOR.quantum_efficiency,
        )
        sky_dark = sky_electrons_per_pixel(
            21.5,
            frame_e,
            OPTICS.aperture_mm / 1000.0,
            1.0 / SENSOR.frame_rate_hz,
            SENSOR.quantum_efficiency,
        )
        assert sky_bright > sky_dark

    def test_empty_frame_mean_near_sky_level(self) -> None:
        """Mean electron count of an empty frame must be near the sky background level.

        Uses data_float (unquantised electrons) so the assertion holds at the
        operational dark-sky level (21 mag/arcsec², ~0.3 e-/px) where uint16
        truncation of negative readout-noise samples would otherwise bias
        frame.data.mean() below sky_e_per_pixel (WP-C3).
        """
        rng = np.random.default_rng(2)
        frame = generate_frame(
            SENSOR, OPTICS, sky_mag_arcsec2=21.0, satellites=[], stars=[], rng=rng
        )
        mean_e = float(frame.data_float.mean())
        assert mean_e == pytest.approx(frame.sky_e_per_pixel, rel=0.05)

    def test_data_float_dtype_float64(self) -> None:
        """data_float must be float64 (unquantised electron domain)."""
        frame = generate_frame(SENSOR, OPTICS, rng=np.random.default_rng(3))
        assert frame.data_float.dtype == np.float64

    def test_data_float_shape_matches_data(self) -> None:
        frame = generate_frame(SENSOR, OPTICS, rng=np.random.default_rng(4))
        assert frame.data_float.shape == frame.data.shape


# ---------------------------------------------------------------------------
# 3. Stellar PSF injection
# ---------------------------------------------------------------------------


class TestStarInjection:
    """Injected stars must be detectable and carry the expected signal."""

    def test_bright_star_above_sky(self) -> None:
        """A V=8 star should produce a local peak well above sky background."""
        cx, cy = 960.0, 540.0
        rng = np.random.default_rng(3)
        frame = generate_frame(
            SENSOR,
            OPTICS,
            sky_mag_arcsec2=21.0,
            stars=[StarSpec(magnitude=8.0, x=cx, y=cy)],
            rng=rng,
        )
        # Local sum in a 5×5 box should greatly exceed background
        box = frame.data[int(cy) - 2 : int(cy) + 3, int(cx) - 2 : int(cx) + 3].astype(
            float
        )
        bg_box = frame.sky_e_per_pixel * 25
        assert box.sum() > bg_box * 5

    def test_star_ground_truth_signal_positive(self) -> None:
        frame = generate_frame(
            SENSOR,
            OPTICS,
            stars=[StarSpec(magnitude=10.0, x=200.0, y=200.0)],
            rng=np.random.default_rng(4),
        )
        assert len(frame.stars) == 1
        assert frame.stars[0].signal_electrons > 0

    def test_star_ground_truth_magnitude_preserved(self) -> None:
        frame = generate_frame(
            SENSOR,
            OPTICS,
            stars=[StarSpec(magnitude=9.5, x=300.0, y=300.0)],
            rng=np.random.default_rng(5),
        )
        assert frame.stars[0].magnitude == pytest.approx(9.5)

    def test_multiple_stars_injected(self) -> None:
        stars = [StarSpec(8.0, 100.0, 100.0), StarSpec(10.0, 500.0, 400.0)]
        frame = generate_frame(
            SENSOR, OPTICS, stars=stars, rng=np.random.default_rng(6)
        )
        assert len(frame.stars) == 2

    def test_brighter_star_larger_signal(self) -> None:
        """V=8 star should inject more electrons than V=12 star."""
        f1 = generate_frame(
            SENSOR,
            OPTICS,
            stars=[StarSpec(8.0, 200.0, 200.0)],
            rng=np.random.default_rng(7),
        )
        f2 = generate_frame(
            SENSOR,
            OPTICS,
            stars=[StarSpec(12.0, 200.0, 200.0)],
            rng=np.random.default_rng(7),
        )
        assert f1.stars[0].signal_electrons > f2.stars[0].signal_electrons


# ---------------------------------------------------------------------------
# 4. Satellite streak injection
# ---------------------------------------------------------------------------


class TestSatelliteInjection:
    """Satellite streaks must carry radiometrically-correct signal."""

    def test_streak_ground_truth_count(self) -> None:
        sats = [
            SatelliteSpec(magnitude=10.0, angular_velocity_deg_s=0.5),
            SatelliteSpec(magnitude=12.0, angular_velocity_deg_s=1.0),
        ]
        frame = generate_frame(
            SENSOR, OPTICS, satellites=sats, rng=np.random.default_rng(8)
        )
        assert len(frame.satellites) == 2

    def test_streak_signal_positive(self) -> None:
        frame = generate_frame(
            SENSOR,
            OPTICS,
            satellites=[SatelliteSpec(magnitude=11.0, angular_velocity_deg_s=0.5)],
            rng=np.random.default_rng(9),
        )
        assert frame.satellites[0].signal_electrons > 0

    def test_brighter_satellite_larger_signal(self) -> None:
        """mv 10 should inject more electrons per pixel than mv 13."""
        f_bright = generate_frame(
            SENSOR,
            OPTICS,
            satellites=[SatelliteSpec(magnitude=10.0, angular_velocity_deg_s=0.5)],
            rng=np.random.default_rng(10),
        )
        f_faint = generate_frame(
            SENSOR,
            OPTICS,
            satellites=[SatelliteSpec(magnitude=13.0, angular_velocity_deg_s=0.5)],
            rng=np.random.default_rng(10),
        )
        assert (
            f_bright.satellites[0].signal_electrons
            > f_faint.satellites[0].signal_electrons
        )

    def test_satellite_signal_matches_radiometry(self) -> None:
        """satellite_signal_electrons must match generate_frame ground truth."""
        mag = 11.0
        ang_vel = 0.5
        expected = satellite_signal_electrons(
            mag, ang_vel, SENSOR, OPTICS, elevation_deg=45.0
        )
        frame = generate_frame(
            SENSOR,
            OPTICS,
            elevation_deg=45.0,
            satellites=[SatelliteSpec(magnitude=mag, angular_velocity_deg_s=ang_vel)],
            rng=np.random.default_rng(11),
        )
        assert frame.satellites[0].signal_electrons == pytest.approx(expected, rel=1e-4)

    def test_streak_start_end_differ(self) -> None:
        """A moving satellite must have different start/end positions."""
        frame = generate_frame(
            SENSOR,
            OPTICS,
            satellites=[SatelliteSpec(magnitude=10.0, angular_velocity_deg_s=1.0)],
            rng=np.random.default_rng(12),
        )
        sat = frame.satellites[0]
        assert sat.x_start != sat.x_end or sat.y_start != sat.y_end

    def test_limiting_magnitude_snr_near_threshold(self) -> None:
        """OpTA.NOD.DET (mv 13.0) should give single-frame SNR < 5 (needs stacking)."""
        sig_e = satellite_signal_electrons(
            13.0, 0.5, SENSOR, OPTICS, elevation_deg=45.0
        )
        pixel_scale = compute_pixel_scale(SENSOR.pixel_size_um, OPTICS.focal_length_mm)
        aperture_m = OPTICS.aperture_mm / 1000.0
        integration_s = 1.0 / SENSOR.frame_rate_hz
        sky_e = sky_electrons_per_pixel(
            21.0, pixel_scale, aperture_m, integration_s, SENSOR.quantum_efficiency
        )
        dark_e = SENSOR.dark_current_e_s * integration_s
        snr = compute_snr(sig_e, sky_e, dark_e, SENSOR.readout_noise_e, n_pixels=5.0)
        # Single frame SNR should be below 5 (stacking needed to reach SNR>=5)
        assert snr < 5.0
        # But signal must be non-zero
        assert sig_e > 0


# ---------------------------------------------------------------------------
# 5. FITS round-trip
# ---------------------------------------------------------------------------


class TestFITSRoundTrip:
    """Write a frame to disk and read it back; data must be preserved exactly."""

    def test_write_and_read_back(self, tmp_path: Path) -> None:
        from astropy.io import fits

        frame = generate_frame(
            SENSOR,
            OPTICS,
            satellites=[SatelliteSpec(10.0, 0.5)],
            stars=[StarSpec(9.0, 480.0, 270.0)],
            rng=np.random.default_rng(13),
        )
        out = write_fits(frame, tmp_path / "test.fits")
        assert out.exists()

        with fits.open(out) as hdl:
            data_back = hdl[0].data
        np.testing.assert_array_equal(frame.data, data_back)

    def test_fits_file_non_empty(self, tmp_path: Path) -> None:
        frame = generate_frame(SENSOR, OPTICS, rng=np.random.default_rng(14))
        out = write_fits(frame, tmp_path / "nonempty.fits")
        assert out.stat().st_size > 1000


# ---------------------------------------------------------------------------
# 6. FITS header completeness
# ---------------------------------------------------------------------------


class TestFITSHeader:
    """FITS header must contain all required fields from pipeline_defaults.yaml."""

    _REQUIRED = {"DATE-OBS", "EXPTIME", "INSTRUME", "GAIN", "TEMP"}

    def test_required_header_fields_present(self) -> None:
        frame = generate_frame(SENSOR, OPTICS, rng=np.random.default_rng(15))
        for key in self._REQUIRED:
            assert key in frame.header or key.replace("-", "_") in frame.header

    def test_exptime_matches_sensor(self) -> None:
        frame = generate_frame(SENSOR, OPTICS, rng=np.random.default_rng(16))
        expected_exptime = 1.0 / SENSOR.frame_rate_hz
        assert frame.header["EXPTIME"] == pytest.approx(expected_exptime)

    def test_node_id_in_header(self) -> None:
        frame = generate_frame(
            SENSOR, OPTICS, node_id="TEST-NODE", rng=np.random.default_rng(17)
        )
        assert frame.header["NODE-ID"] == "TEST-NODE"
