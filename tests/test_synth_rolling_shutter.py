"""Tests for rolling-shutter readout injection (T-03 forward model).

Two layers:

1. **Model** — :class:`RollingShutterReadout.displace`: identity at zero
   readout, identity for a stationary source, identity at the reference row,
   and a displacement proportional to velocity × (row − reference) × t_row.

2. **Generator integration** — a *moving* satellite rendered with the readout
   model is displaced along its velocity by the per-row timing bias, while a
   *stationary* star is byte-identical with and without it.  This is the
   ground-truth injection the pipeline's T-03 correction must recover.
"""

from __future__ import annotations

import numpy as np
import pytest
from sensor_fixtures import OPTICS_DEFAULT, SENSOR_SMALL

from opta_pipeline.synth import (
    SatelliteSpec,
    StarSpec,
    generate_frame,
)
from opta_pipeline.synth.rolling_shutter import RollingShutterReadout

SENSOR = SENSOR_SMALL
OPTICS = OPTICS_DEFAULT
_SHAPE = (SENSOR.resolution_v, SENSOR.resolution_h)
_REF = (SENSOR.resolution_v - 1) / 2.0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TestRollingShutterModel:
    def test_zero_readout_is_identity(self) -> None:
        rs = RollingShutterReadout(row_readout_us=0.0)
        assert rs.displace(120.0, 40.0, 100.0, 50.0, _SHAPE) == (120.0, 40.0)

    def test_stationary_source_is_identity(self) -> None:
        rs = RollingShutterReadout(row_readout_us=130.0)
        assert rs.displace(120.0, 40.0, 0.0, 0.0, _SHAPE) == (120.0, 40.0)

    def test_reference_row_is_identity(self) -> None:
        """A source on the reference row has zero timing offset."""
        rs = RollingShutterReadout(row_readout_us=130.0, reference_row=150.0)
        assert rs.displace(200.0, 150.0, 300.0, 0.0, _SHAPE) == (200.0, 150.0)

    def test_displacement_matches_formula(self) -> None:
        rs = RollingShutterReadout(row_readout_us=130.0, reference_row=150.0)
        vx, vy, y = 300.0, 80.0, 250.0
        xd, yd = rs.displace(200.0, y, vx, vy, _SHAPE)
        dt = (y - 150.0) * 130e-6
        assert xd == pytest.approx(200.0 + vx * dt)
        assert yd == pytest.approx(y + vy * dt)

    def test_displacement_scales_with_velocity(self) -> None:
        rs = RollingShutterReadout(row_readout_us=130.0, reference_row=150.0)
        x1, _ = rs.displace(200.0, 250.0, 100.0, 0.0, _SHAPE)
        x2, _ = rs.displace(200.0, 250.0, 200.0, 0.0, _SHAPE)
        assert (x2 - 200.0) == pytest.approx(2.0 * (x1 - 200.0))

    def test_default_reference_is_frame_centre(self) -> None:
        rs = RollingShutterReadout(row_readout_us=130.0)  # reference_row=None
        # A source at the frame-centre row is undisplaced.
        assert rs.displace(200.0, _REF, 300.0, 0.0, _SHAPE) == (200.0, _REF)


# ---------------------------------------------------------------------------
# Generator integration
# ---------------------------------------------------------------------------


def _flux_centroid(frame: np.ndarray) -> tuple[float, float]:
    """Flux-weighted centroid of the brightest source in a near-empty frame.

    Thresholds at 20 % of the peak so the diffuse noise floor (positive after
    background subtraction across every pixel) cannot dilute the streak.
    """
    a = frame.astype(np.float64) - np.median(frame)
    a = np.where(a > 0.2 * a.max(), a, 0.0)
    ys, xs = np.indices(a.shape)
    tot = a.sum()
    return float((xs * a).sum() / tot), float((ys * a).sum() / tot)


class TestRollingShutterInFrame:
    def test_none_is_byte_identical(self) -> None:
        sat = SatelliteSpec(magnitude=6.0, angular_velocity_deg_s=1.0,
                            x_center=200.0, y_center=250.0, angle_deg=0.0)
        a = generate_frame(SENSOR, OPTICS, satellites=[sat],
                           rng=np.random.default_rng(0))
        b = generate_frame(SENSOR, OPTICS, satellites=[sat], rolling_shutter=None,
                           rng=np.random.default_rng(0))
        assert np.array_equal(a.data_float, b.data_float)

    def test_stationary_star_unaffected(self) -> None:
        """A star has zero velocity, so readout leaves it byte-identical."""
        rs = RollingShutterReadout(row_readout_us=200.0, reference_row=150.0)
        star = StarSpec(7.0, 300.0, 40.0)
        plain = generate_frame(SENSOR, OPTICS, stars=[star],
                               rng=np.random.default_rng(1))
        rolled = generate_frame(SENSOR, OPTICS, stars=[star], rolling_shutter=rs,
                                rng=np.random.default_rng(1))
        assert np.array_equal(plain.data_float, rolled.data_float)

    def test_moving_source_displaced_along_velocity(self) -> None:
        """A horizontal streak off the reference row shifts in x by ~v·Δt."""
        ref = 150.0
        y0 = 250.0  # 100 rows below the reference row
        row_us = 200.0
        sat = SatelliteSpec(magnitude=5.0, angular_velocity_deg_s=2.0,
                            x_center=200.0, y_center=y0, angle_deg=0.0)
        rs = RollingShutterReadout(row_readout_us=row_us, reference_row=ref)
        plain = generate_frame(SENSOR, OPTICS, satellites=[sat],
                               rng=np.random.default_rng(2))
        rolled = generate_frame(SENSOR, OPTICS, satellites=[sat], rolling_shutter=rs,
                                rng=np.random.default_rng(2))
        cx_plain, _ = _flux_centroid(plain.data_float)
        cx_rolled, _ = _flux_centroid(rolled.data_float)

        v_px_s = 2.0 * 3600.0 / plain.pixel_scale_arcsec
        expected_dx = v_px_s * (y0 - ref) * row_us * 1e-6
        assert (cx_rolled - cx_plain) == pytest.approx(expected_dx, rel=0.1, abs=0.2)
        assert expected_dx > 1.0  # a meaningful, detectable shift
