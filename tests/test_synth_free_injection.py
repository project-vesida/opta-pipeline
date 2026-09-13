"""Tests for the free-injection (non-physics-traceable) source mode.

Setting ``signal_e`` on a StarSpec/SatelliteSpec injects the source at an exact
electron count, bypassing ``opta_model.radiometry`` entirely.  This is the
stress-testing path: it lets a test place a source of known brightness/morphology
without coupling to the radiometric chain.  The default (``signal_e=None``,
magnitude-driven) stays physics-traceable and unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
from sensor_fixtures import OPTICS_DEFAULT, SENSOR_SMALL

from opta_pipeline.synth import (
    SatelliteSpec,
    StarSpec,
    build_detector_model,
    generate_frame,
)

SENSOR = SENSOR_SMALL
OPTICS = OPTICS_DEFAULT


class TestFreeInjectionClean:
    """Clean path: signal_e sets the deposited flux exactly."""

    def test_star_flux_equals_signal_e(self) -> None:
        f = generate_frame(
            SENSOR, OPTICS,
            stars=[StarSpec(magnitude=99.0, x=200, y=150, signal_e=50000.0)],
            rng=np.random.default_rng(0),
        )
        assert f.stars[0].signal_electrons == 50000.0
        # Deposited flux above the sky background matches the requested total.
        bg = f.sky_e_per_pixel
        box = f.data_float[135:166, 185:216].sum() - bg * 31 * 31
        assert box == pytest.approx(50000.0, rel=0.02)

    def test_magnitude_ignored_when_signal_e_set(self) -> None:
        """Two wildly different magnitudes + same signal_e → identical frames."""
        a = generate_frame(
            SENSOR, OPTICS, stars=[StarSpec(5.0, 200, 150, signal_e=30000.0)],
            rng=np.random.default_rng(1),
        )
        b = generate_frame(
            SENSOR, OPTICS, stars=[StarSpec(15.0, 200, 150, signal_e=30000.0)],
            rng=np.random.default_rng(1),
        )
        assert np.array_equal(a.data_float, b.data_float)

    def test_satellite_total_streak_flux(self) -> None:
        f = generate_frame(
            SENSOR, OPTICS,
            satellites=[SatelliteSpec(
                magnitude=99.0, angular_velocity_deg_s=0.5,
                x_center=200, y_center=150, signal_e=20000.0,
            )],
            rng=np.random.default_rng(0),
        )
        bg = f.sky_e_per_pixel
        total = f.data_float.sum() - bg * SENSOR.resolution_h * SENSOR.resolution_v
        assert total == pytest.approx(20000.0, rel=0.02)
        # Per-pixel ground truth is the total spread over the trail length.
        assert 0.0 < f.satellites[0].signal_electrons < 20000.0

    def test_traceable_default_unaffected(self) -> None:
        """signal_e=None keeps the magnitude-driven (radiometric) behaviour."""
        assert StarSpec(8.0, 200, 150).signal_e is None  # default stays traceable
        bright = generate_frame(
            SENSOR, OPTICS, stars=[StarSpec(8.0, 200, 150)],
            rng=np.random.default_rng(2),
        )
        faint = generate_frame(
            SENSOR, OPTICS, stars=[StarSpec(12.0, 200, 150)],
            rng=np.random.default_rng(2),
        )
        # Radiometry still governs: a brighter magnitude → more signal.
        assert bright.stars[0].signal_electrons > faint.stars[0].signal_electrons > 0.0


class TestFreeInjectionRealistic:
    """Realistic path: signal_e flows through the detector chain too."""

    def test_star_flux_equals_signal_e(self) -> None:
        det = build_detector_model(
            SENSOR, seed=1, bias_offset_e=0.0, bias_fpn_rms_e=0.0, prnu_pct=0.0,
            hot_pixel_fraction=0.0, dead_pixel_fraction=0.0,
            vignetting_corner_factor=1.0,
        )
        f = generate_frame(
            SENSOR, OPTICS,
            stars=[StarSpec(magnitude=99.0, x=200, y=150, signal_e=40000.0)],
            detector=det, rng=np.random.default_rng(0),
        )
        assert f.stars[0].signal_electrons == 40000.0
        bg = f.sky_e_per_pixel
        box = f.data_float[135:166, 185:216].sum() - bg * 31 * 31
        assert box == pytest.approx(40000.0, rel=0.05)

    def test_independent_of_radiometry(self) -> None:
        """Same signal_e at different elevations → same source flux (no extinction
        applied to a free-injected source)."""
        det = build_detector_model(
            SENSOR, seed=2, bias_offset_e=0.0, bias_fpn_rms_e=0.0, prnu_pct=0.0,
            hot_pixel_fraction=0.0, vignetting_corner_factor=1.0,
        )
        kw = dict(
            stars=[StarSpec(10.0, 200, 150, signal_e=25000.0)],
            detector=det,
        )
        hi = generate_frame(SENSOR, OPTICS, elevation_deg=80.0, **kw,
                            rng=np.random.default_rng(5))
        lo = generate_frame(SENSOR, OPTICS, elevation_deg=20.0, **kw,
                            rng=np.random.default_rng(5))
        # Source core flux identical (only sky differs slightly by elevation,
        # but with this clean detector and same rng the star box is unchanged).
        box_hi = hi.data_float[145:156, 195:206].sum()
        box_lo = lo.data_float[145:156, 195:206].sum()
        assert box_hi == pytest.approx(box_lo, rel=1e-6)
