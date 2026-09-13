"""I-02 output schema validation tests (WP-C5).

Verifies that tracklets_to_records() produces records that match the I-02
interface specification from SYSTEMS.md: every required field present, correct
Python type, and physically reasonable value range.

No jsonschema dependency — validated with explicit type and range checks so
the test suite stays self-contained.
"""

from __future__ import annotations

import math

import pytest
from opta_model.hardware import VILTROX_85_F14_PRESET
from sensor_fixtures import SENSOR_SMALL

from opta_pipeline.astrometry import (
    StarMatch,
    WCSSolution,
    astrometrise_detections,
    fit_wcs,
)
from opta_pipeline.detect import Detection
from opta_pipeline.tracklet import (
    FrameDetections,
    Tracklet,
    link_detections,
    tracklets_to_json,
    tracklets_to_records,
)

_OPTICS = VILTROX_85_F14_PRESET

# I-02 required fields (SYSTEMS.md interface spec)
_I02_REQUIRED = {"ra", "dec", "utc", "sigma_ra", "sigma_dec", "object_id", "node_id"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wcs() -> WCSSolution:
    """Minimal 4-star TAN WCS for testing."""
    cx, cy = 200.0, 150.0
    # Build 4 StarMatches on a small grid around frame centre
    from opta_model.hardware import compute_pixel_scale

    ps = compute_pixel_scale(SENSOR_SMALL.pixel_size_um, _OPTICS.focal_length_mm)
    ps_deg = ps / 3600.0
    ra0, dec0 = 135.0, 45.0
    cos_dec = math.cos(math.radians(dec0))
    matches = [
        StarMatch(
            x_px=cx - 50,
            y_px=cy - 40,
            ra_deg=ra0 + 50 * ps_deg / cos_dec,
            dec_deg=dec0 - 40 * ps_deg,
        ),
        StarMatch(
            x_px=cx + 50,
            y_px=cy - 40,
            ra_deg=ra0 - 50 * ps_deg / cos_dec,
            dec_deg=dec0 - 40 * ps_deg,
        ),
        StarMatch(
            x_px=cx - 50,
            y_px=cy + 40,
            ra_deg=ra0 + 50 * ps_deg / cos_dec,
            dec_deg=dec0 + 40 * ps_deg,
        ),
        StarMatch(
            x_px=cx + 50,
            y_px=cy + 40,
            ra_deg=ra0 - 50 * ps_deg / cos_dec,
            dec_deg=dec0 + 40 * ps_deg,
        ),
    ]
    return fit_wcs(matches, frame_shape=(300, 400))


def _make_tracklet(n_points: int = 5, node_id: str = "NODE-01") -> Tracklet:
    """Create a minimal Tracklet with n_points observations."""
    wcs = _make_wcs()
    det = Detection(
        x=200.0,
        y=150.0,
        snr=30.0,
        elongation=3.0,
        angle_deg=0.0,
        n_pixels=20,
        flux_e=5000.0,
        is_streak=True,
        fwhm_px=3.0,
    )
    astrometrise_detections([det], wcs)
    frame_dets = []
    for i in range(n_points):
        # Shift the detection slightly each frame to simulate motion
        shifted = Detection(
            x=200.0 + i * 8.0,
            y=150.0,
            snr=30.0,
            elongation=3.0,
            angle_deg=0.0,
            n_pixels=20,
            flux_e=5000.0,
            is_streak=True,
            fwhm_px=3.0,
        )
        astro_i = astrometrise_detections([shifted], wcs)
        frame_dets.append(
            FrameDetections(
                detections=tuple(astro_i),
                utc_mjd=60000.0 + i * (0.04 / 86400.0),
                frame_id=i,
                node_id=node_id,
            )
        )
    tracklets = link_detections(
        frame_dets,
        max_sep_arcsec=500.0,
        min_points=3,
        max_rms_arcsec=200.0,
        max_gap_frames=1,
    )
    assert tracklets, "Test setup failed: no tracklet formed"
    return tracklets[0]


# ---------------------------------------------------------------------------
# 1. Required field presence
# ---------------------------------------------------------------------------


class TestI02RequiredFields:
    """Every I-02 record must contain all required fields."""

    def test_all_required_fields_present(self) -> None:
        t = _make_tracklet()
        for rec in t.to_records():
            missing = _I02_REQUIRED - set(rec.keys())
            assert not missing, f"Missing I-02 fields: {missing}"

    def test_tracklets_to_records_all_fields(self) -> None:
        t = _make_tracklet()
        records = tracklets_to_records([t])
        assert len(records) > 0
        for rec in records:
            assert _I02_REQUIRED <= set(rec.keys())

    def test_no_extra_undocumented_fields(self) -> None:
        """Records should not carry fields beyond the I-02 spec."""
        t = _make_tracklet()
        for rec in t.to_records():
            extra = set(rec.keys()) - _I02_REQUIRED
            assert not extra, f"Unexpected I-02 fields: {extra}"


# ---------------------------------------------------------------------------
# 2. Field types
# ---------------------------------------------------------------------------


class TestI02FieldTypes:
    """I-02 field types match the interface spec."""

    def test_ra_dec_are_float(self) -> None:
        t = _make_tracklet()
        for rec in t.to_records():
            assert isinstance(rec["ra"], float)
            assert isinstance(rec["dec"], float)

    def test_utc_is_float(self) -> None:
        t = _make_tracklet()
        for rec in t.to_records():
            assert isinstance(rec["utc"], float)

    def test_sigma_fields_are_float(self) -> None:
        t = _make_tracklet()
        for rec in t.to_records():
            assert isinstance(rec["sigma_ra"], float)
            assert isinstance(rec["sigma_dec"], float)

    def test_object_id_is_str(self) -> None:
        t = _make_tracklet()
        for rec in t.to_records():
            assert isinstance(rec["object_id"], str)

    def test_node_id_is_str(self) -> None:
        t = _make_tracklet()
        for rec in t.to_records():
            assert isinstance(rec["node_id"], str)


# ---------------------------------------------------------------------------
# 3. Value ranges
# ---------------------------------------------------------------------------


class TestI02ValueRanges:
    """I-02 values must be physically reasonable."""

    def test_ra_in_range(self) -> None:
        t = _make_tracklet()
        for rec in t.to_records():
            assert 0.0 <= rec["ra"] < 360.0, f"RA out of range: {rec['ra']}"

    def test_dec_in_range(self) -> None:
        t = _make_tracklet()
        for rec in t.to_records():
            assert -90.0 <= rec["dec"] <= 90.0, f"Dec out of range: {rec['dec']}"

    def test_sigma_positive(self) -> None:
        t = _make_tracklet()
        for rec in t.to_records():
            assert rec["sigma_ra"] > 0.0
            assert rec["sigma_dec"] > 0.0

    def test_utc_recent_mjd(self) -> None:
        """MJD should be a reasonable value (post-J2000 = 51544)."""
        t = _make_tracklet()
        for rec in t.to_records():
            assert rec["utc"] > 51544.0, f"MJD looks pre-J2000: {rec['utc']}"

    def test_node_id_preserved(self) -> None:
        t = _make_tracklet(node_id="NODE-TEST")
        for rec in t.to_records():
            assert rec["node_id"] == "NODE-TEST"


# ---------------------------------------------------------------------------
# 4. JSON serialisation
# ---------------------------------------------------------------------------


class TestI02JSONOutput:
    """tracklets_to_json produces valid JSON with the correct structure."""

    def test_json_is_valid(self) -> None:
        import json

        t = _make_tracklet()
        text = tracklets_to_json([t])
        parsed = json.loads(text)
        assert isinstance(parsed, list)
        assert len(parsed) > 0

    def test_json_roundtrip_ra(self) -> None:
        import json

        t = _make_tracklet()
        text = tracklets_to_json([t])
        parsed = json.loads(text)
        records = t.to_records()
        for parsed_rec, orig_rec in zip(parsed, records):
            assert parsed_rec["ra"] == pytest.approx(orig_rec["ra"])
