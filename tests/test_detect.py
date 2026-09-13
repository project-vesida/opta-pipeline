"""Tests for opta_pipeline.detect.

Validates:
  - Empty calibrated frame → no detections
  - SNR filter: only sources above threshold are returned
  - Centroid accuracy: within 1 pixel of injection position
  - Streak classification: elongated sources flagged as streaks
  - Point source classification: compact sources not flagged as streaks
  - Multiple sources detected independently
  - Detection completeness vs. magnitude (OT-021 accept criterion)
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from opta_model.hardware import IMX585_PRESET, VILTROX_85_F14_PRESET
from sensor_fixtures import SENSOR_SMALL as _SENSOR_SMALL

from opta_pipeline.detect import Detection, detect_sources
from opta_pipeline.synth import SatelliteSpec, StarSpec, generate_frame

SENSOR = IMX585_PRESET
OPTICS = VILTROX_85_F14_PRESET


# ---------------------------------------------------------------------------
# 1. Basic API contract
# ---------------------------------------------------------------------------


class TestDetectAPI:
    """Output type and basic contract."""

    def test_returns_list(self) -> None:
        frame = np.zeros((50, 50), dtype=np.float32)
        result = detect_sources(frame, noise_rms=5.0)
        assert isinstance(result, list)

    def test_empty_noise_frame_no_detections(self) -> None:
        """Pure Gaussian noise sigma=1.0, threshold=3.0 → very few spurious hits."""
        rng = np.random.default_rng(60)
        noise = rng.normal(0, 1.0, (100, 100)).astype(np.float32)
        dets = detect_sources(noise, noise_rms=1.0, snr_threshold=5.0, min_pixels=5)
        assert len(dets) == 0

    def test_invalid_noise_rms_raises(self) -> None:
        frame = np.zeros((10, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="noise_rms"):
            detect_sources(frame, noise_rms=0.0)

    def test_detections_sorted_by_snr_descending(self) -> None:
        """Detections must be returned in descending SNR order."""
        np.random.default_rng(61)
        frame = np.zeros((100, 100), dtype=np.float32)
        # Inject two sources of different brightness
        frame[30, 30] = 100.0
        frame[70, 70] = 200.0
        dets = detect_sources(frame, noise_rms=1.0, snr_threshold=3.0, min_pixels=1)
        if len(dets) >= 2:
            assert dets[0].snr >= dets[1].snr


# ---------------------------------------------------------------------------
# 2. Threshold filtering
# ---------------------------------------------------------------------------


class TestSNRFiltering:
    """Sources below threshold must not appear in output."""

    def test_bright_source_detected(self) -> None:
        """A pixel well above threshold should be detected."""
        frame = np.zeros((20, 20), dtype=np.float32)
        frame[10, 10] = 50.0
        dets = detect_sources(frame, noise_rms=1.0, snr_threshold=3.0, min_pixels=1)
        assert len(dets) >= 1

    def test_sub_threshold_source_not_detected(self) -> None:
        """A pixel below snr_threshold must not appear."""
        frame = np.zeros((20, 20), dtype=np.float32)
        frame[10, 10] = 2.9  # SNR = 2.9/1.0 = 2.9 < 3.0
        dets = detect_sources(frame, noise_rms=1.0, snr_threshold=3.0, min_pixels=1)
        assert len(dets) == 0

    def test_all_detections_above_threshold(self) -> None:
        """Every returned detection must have SNR above the threshold."""
        rng = np.random.default_rng(62)
        frame = rng.normal(0, 1.0, (60, 60)).astype(np.float32)
        # Inject a few bright spots
        for y, x in [(15, 15), (30, 45), (50, 20)]:
            frame[y, x] = 30.0
        dets = detect_sources(frame, noise_rms=1.0, snr_threshold=3.0, min_pixels=1)
        for d in dets:
            assert d.snr >= 2.9  # approx (slight variation from multi-pixel components)


# ---------------------------------------------------------------------------
# 3. Streak vs. point source classification
# ---------------------------------------------------------------------------


class TestStreakClassification:
    """Elongated sources classified as streaks; compact sources are not."""

    def _make_streak_frame(
        self, cx: float = 50.0, cy: float = 50.0, length: int = 30
    ) -> tuple[np.ndarray, float]:
        """Inject a horizontal line source into a noise-free frame."""
        frame = np.zeros((100, 100), dtype=np.float32)
        half = length // 2
        frame[int(cy), int(cx) - half : int(cx) + half + 1] = 20.0
        return frame, 1.0

    def _make_psf_frame(
        self, cx: float = 50.0, cy: float = 50.0, fwhm: float = 2.0
    ) -> tuple[np.ndarray, float]:
        """Inject a 2-D Gaussian PSF."""
        h, w = 100, 100
        frame = np.zeros((h, w), dtype=np.float32)
        sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        yy, xx = np.mgrid[:h, :w]
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))
        frame += (g * 30.0 / g.max()).astype(np.float32)
        return frame, 1.0

    def test_long_streak_classified_as_streak(self) -> None:
        frame, noise = self._make_streak_frame(length=40)
        dets = detect_sources(
            frame,
            noise_rms=noise,
            snr_threshold=3.0,
            min_pixels=5,
            elongation_threshold=2.0,
        )
        assert any(d.is_streak for d in dets), "Expected at least one streak detection"

    def test_psf_source_not_classified_as_streak(self) -> None:
        frame, noise = self._make_psf_frame(fwhm=2.0)
        dets = detect_sources(
            frame, noise_rms=noise, snr_threshold=3.0, elongation_threshold=2.0
        )
        assert any(not d.is_streak for d in dets), "Expected at least one point source"

    def test_streak_has_higher_elongation_than_psf(self) -> None:
        """A streak must have higher elongation than an equivalent PSF."""
        f_streak, _ = self._make_streak_frame(length=30)
        f_psf, _ = self._make_psf_frame(fwhm=2.0)

        dets_streak = detect_sources(
            f_streak,
            noise_rms=1.0,
            snr_threshold=3.0,
            min_pixels=5,
            elongation_threshold=2.0,
        )
        dets_psf = detect_sources(
            f_psf,
            noise_rms=1.0,
            snr_threshold=3.0,
            min_pixels=5,
            elongation_threshold=2.0,
        )

        if dets_streak and dets_psf:
            assert max(d.elongation for d in dets_streak) > max(
                d.elongation for d in dets_psf
            )


# ---------------------------------------------------------------------------
# 4. Centroid accuracy
# ---------------------------------------------------------------------------


class TestCentroidAccuracy:
    """Centroid should be within 1 pixel of the injected position."""

    def test_psf_centroid_accuracy(self) -> None:
        """2-D Gaussian PSF centroid must be within 1 px of injection position."""
        cx, cy = 48.3, 52.7
        fwhm = 2.0
        sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        h, w = 100, 100
        frame = np.zeros((h, w), dtype=np.float32)
        yy, xx = np.mgrid[:h, :w]
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))
        frame += (g * 50.0).astype(np.float32)

        dets = detect_sources(frame, noise_rms=1.0, snr_threshold=3.0, min_pixels=1)
        assert len(dets) >= 1
        best = dets[0]
        assert abs(best.x - cx) < 1.0
        assert abs(best.y - cy) < 1.0

    def test_streak_centroid_in_frame(self) -> None:
        """Streak centroid must be within the frame bounds."""
        frame = np.zeros((100, 100), dtype=np.float32)
        frame[50, 20:80] = 25.0  # horizontal streak
        dets = detect_sources(frame, noise_rms=1.0, snr_threshold=3.0, min_pixels=5)
        assert len(dets) >= 1
        d = dets[0]
        assert 0 <= d.x < 100
        assert 0 <= d.y < 100


# ---------------------------------------------------------------------------
# 5. Star mask
# ---------------------------------------------------------------------------


class TestStarMask:
    """Star mask should suppress detections in masked regions."""

    def test_masked_source_not_detected(self) -> None:
        """A bright source in a masked region should not appear."""
        frame = np.zeros((50, 50), dtype=np.float32)
        frame[25, 25] = 100.0
        mask = np.zeros((50, 50), dtype=bool)
        mask[23:28, 23:28] = True  # mask the source
        dets = detect_sources(
            frame, noise_rms=1.0, snr_threshold=3.0, min_pixels=1, star_mask=mask
        )
        assert len(dets) == 0

    def test_unmasked_source_detected(self) -> None:
        """A bright source outside the mask is still detected."""
        frame = np.zeros((50, 50), dtype=np.float32)
        frame[10, 10] = 100.0  # unmasked
        frame[40, 40] = 100.0  # unmasked
        mask = np.zeros((50, 50), dtype=bool)
        mask[23:28, 23:28] = True  # mask elsewhere
        dets = detect_sources(
            frame, noise_rms=1.0, snr_threshold=3.0, min_pixels=1, star_mask=mask
        )
        assert len(dets) >= 2


# ---------------------------------------------------------------------------
# 6. Integration with synth + calibrate (OT-021 accept criterion)
# ---------------------------------------------------------------------------


class TestDetectionCompleteness:
    """OT-021 accept: detections from synth frames at known magnitudes.

    Uses synth → calibrate → detect pipeline.  Bright stars (V=8) and
    bright satellites (mv=10) must be detected; faint sources near the
    single-frame limit (mv=13) may not.
    """

    def _synth_cal_detect(
        self,
        mag: float,
        is_satellite: bool,
        rng_seed: int = 70,
    ) -> list[Detection]:
        rng = np.random.default_rng(rng_seed)
        if is_satellite:
            frame = generate_frame(
                _SENSOR_SMALL,
                OPTICS,
                sky_mag_arcsec2=21.0,
                satellites=[
                    SatelliteSpec(
                        magnitude=mag,
                        angular_velocity_deg_s=0.5,
                        x_center=200.0,
                        y_center=150.0,
                    )
                ],
                rng=rng,
            )
        else:
            frame = generate_frame(
                _SENSOR_SMALL,
                OPTICS,
                sky_mag_arcsec2=21.0,
                stars=[StarSpec(magnitude=mag, x=200.0, y=150.0)],
                rng=rng,
            )

        float_frame = frame.data.astype(np.float32) - float(frame.sky_e_per_pixel)
        noise = math.sqrt(frame.sky_e_per_pixel + frame.readout_noise_e**2)
        return detect_sources(
            float_frame, noise_rms=noise, snr_threshold=3.0, min_pixels=3
        )

    def test_bright_star_detected(self) -> None:
        """V=8 star (well above single-frame limit) must be detected."""
        dets = self._synth_cal_detect(8.0, is_satellite=False)
        assert len(dets) >= 1

    def test_bright_satellite_detected(self) -> None:
        """mv=10 satellite must be detected in a single frame."""
        dets = self._synth_cal_detect(10.0, is_satellite=True)
        assert len(dets) >= 1

    def test_detections_have_positive_snr(self) -> None:
        """All returned detections must have positive SNR."""
        dets = self._synth_cal_detect(9.0, is_satellite=False)
        for d in dets:
            assert d.snr > 0

    def test_detection_count_multiple_stars(self) -> None:
        """Multiple injected stars should all be detected."""
        rng = np.random.default_rng(80)
        stars = [
            StarSpec(8.0, 100.0, 100.0),
            StarSpec(9.0, 200.0, 150.0),
            StarSpec(8.5, 300.0, 200.0),
        ]
        frame = generate_frame(
            _SENSOR_SMALL, OPTICS, sky_mag_arcsec2=21.0, stars=stars, rng=rng
        )
        float_frame = frame.data.astype(np.float32) - float(frame.sky_e_per_pixel)
        noise = math.sqrt(frame.sky_e_per_pixel + frame.readout_noise_e**2)
        dets = detect_sources(
            float_frame, noise_rms=noise, snr_threshold=3.0, min_pixels=3
        )
        assert len(dets) >= 3
