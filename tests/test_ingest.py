"""Tests for opta_pipeline.ingest — FITS ingestion and header validation (WP-C6).

Covers:
  - Plain FITS round-trip: write → load → data preserved
  - Rice-compressed FITS round-trip (fpack/astropy CompImageHDU)
  - Header validation: all required fields accepted
  - Missing-header rejection with informative error
  - Malformed-header handling (unreadable file, non-existent file)
  - FrameHeader field types and values
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from sensor_fixtures import OPTICS_DEFAULT, SENSOR_SMALL

from opta_pipeline.ingest import (
    FrameHeader,
    FrameIngestError,
    load_frame,
    validate_header,
)
from opta_pipeline.synth import generate_frame, write_fits

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_REQUIRED_HEADER_VALS = {
    "DATE-OBS": "2026-06-02T12:00:00.000",
    "NODE-ID": "NODE-01",
    "EXPTIME": 0.04,
    "INSTRUME": "IMX585",
    "GAIN": 1.0,
    "TEMP": 20.0,
}


def _write_plain_fits(
    path, data: np.ndarray, extra_headers: dict | None = None
) -> None:
    """Write a plain FITS file with all required headers."""
    hdu = fits.PrimaryHDU(data=data.astype(np.uint16))
    for k, v in _REQUIRED_HEADER_VALS.items():
        hdu.header[k] = v
    if extra_headers:
        for k, v in extra_headers.items():
            hdu.header[k] = v
    hdu.writeto(path, overwrite=True)


def _write_rice_fits(path, data: np.ndarray) -> None:
    """Write a Rice-compressed FITS file with all required headers."""
    image_data = data.astype(np.int32)
    primary = fits.PrimaryHDU()
    for k, v in _REQUIRED_HEADER_VALS.items():
        primary.header[k] = v
    comp_hdu = fits.CompImageHDU(
        data=image_data,
        compression_type="RICE_1",
    )
    hdul = fits.HDUList([primary, comp_hdu])
    hdul.writeto(path, overwrite=True)


# ---------------------------------------------------------------------------
# 1. Plain FITS round-trip
# ---------------------------------------------------------------------------


class TestPlainFITS:
    """Plain FITS write → load → data preserved."""

    def test_roundtrip_pixel_values(self, tmp_path) -> None:
        rng = np.random.default_rng(10)
        original = rng.integers(0, 65535, (100, 100), dtype=np.uint16)
        p = tmp_path / "test.fits"
        _write_plain_fits(p, original)
        _, loaded = load_frame(p)
        np.testing.assert_array_equal(loaded, original)

    def test_roundtrip_shape(self, tmp_path) -> None:
        data = np.zeros((300, 400), dtype=np.uint16)
        p = tmp_path / "shape.fits"
        _write_plain_fits(p, data)
        _, loaded = load_frame(p)
        assert loaded.shape == (300, 400)

    def test_roundtrip_synth_frame(self, tmp_path) -> None:
        """write_fits → load_frame preserves pixel data for a synth frame."""
        rng = np.random.default_rng(11)
        synth = generate_frame(SENSOR_SMALL, OPTICS_DEFAULT, rng=rng)
        p = tmp_path / "synth.fits"
        write_fits(synth, p)
        _, loaded = load_frame(p)
        np.testing.assert_array_equal(loaded, synth.data)

    def test_header_fields_populated(self, tmp_path) -> None:
        data = np.zeros((50, 50), dtype=np.uint16)
        p = tmp_path / "h.fits"
        _write_plain_fits(p, data)
        header, _ = load_frame(p)
        assert isinstance(header, FrameHeader)
        assert header.node_id == "NODE-01"
        assert header.exptime_s == pytest.approx(0.04)
        assert header.gain == pytest.approx(1.0)
        assert header.temp_c == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# 2. Rice-compressed FITS round-trip
# ---------------------------------------------------------------------------


class TestRiceFITS:
    """Rice-compressed FITS must round-trip correctly."""

    def test_rice_roundtrip_pixel_values(self, tmp_path) -> None:
        rng = np.random.default_rng(20)
        original = rng.integers(0, 5000, (100, 100), dtype=np.uint16)
        p = tmp_path / "rice.fits.fz"
        _write_rice_fits(p, original)
        _, loaded = load_frame(p)
        # Rice is lossless for integer data
        np.testing.assert_array_equal(loaded, original)

    def test_rice_roundtrip_shape(self, tmp_path) -> None:
        data = np.zeros((64, 80), dtype=np.uint16)
        p = tmp_path / "rice_shape.fits.fz"
        _write_rice_fits(p, data)
        _, loaded = load_frame(p)
        assert loaded.shape == (64, 80)

    def test_rice_header_fields_present(self, tmp_path) -> None:
        data = np.zeros((32, 32), dtype=np.uint16)
        p = tmp_path / "rice_hdr.fits.fz"
        _write_rice_fits(p, data)
        header, _ = load_frame(p)
        assert header.instrume == "IMX585"
        assert header.node_id == "NODE-01"


# ---------------------------------------------------------------------------
# 3. Header validation
# ---------------------------------------------------------------------------


class TestHeaderValidation:
    """validate_header and load_frame reject incomplete headers."""

    def test_valid_header_passes(self) -> None:
        h = fits.Header()
        for k, v in _REQUIRED_HEADER_VALS.items():
            h[k] = v
        validate_header(h)  # should not raise

    def test_missing_one_header_raises(self) -> None:
        h = fits.Header()
        for k, v in _REQUIRED_HEADER_VALS.items():
            h[k] = v
        del h["NODE-ID"]
        with pytest.raises(FrameIngestError, match="NODE-ID"):
            validate_header(h)

    def test_missing_multiple_headers_raises(self) -> None:
        h = fits.Header()
        h["DATE-OBS"] = "2026-01-01T00:00:00"
        # Missing: NODE-ID, EXPTIME, INSTRUME, GAIN, TEMP
        with pytest.raises(FrameIngestError):
            validate_header(h)

    def test_empty_header_raises(self) -> None:
        with pytest.raises(FrameIngestError):
            validate_header(fits.Header())

    def test_load_frame_missing_header_raises(self, tmp_path) -> None:
        data = np.zeros((20, 20), dtype=np.uint16)
        p = tmp_path / "no_node.fits"
        hdu = fits.PrimaryHDU(data=data)
        hdu.header["DATE-OBS"] = "2026-01-01T00:00:00"
        hdu.header["EXPTIME"] = 0.04
        hdu.header["INSTRUME"] = "IMX585"
        hdu.header["GAIN"] = 1.0
        hdu.header["TEMP"] = 20.0
        # NODE-ID intentionally absent
        hdu.writeto(p, overwrite=True)
        with pytest.raises(FrameIngestError, match="NODE-ID"):
            load_frame(p)

    def test_custom_required_list(self) -> None:
        h = fits.Header()
        h["MY-KEY"] = "present"
        validate_header(h, required=["MY-KEY"])  # should pass

        with pytest.raises(FrameIngestError, match="OTHER-KEY"):
            validate_header(h, required=["OTHER-KEY"])


# ---------------------------------------------------------------------------
# 4. Error cases
# ---------------------------------------------------------------------------


class TestIngestErrors:
    """Malformed or missing files raise informative errors."""

    def test_file_not_found(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_frame(tmp_path / "nonexistent.fits")

    def test_not_a_fits_file(self, tmp_path) -> None:
        p = tmp_path / "garbage.fits"
        p.write_bytes(b"this is not a FITS file")
        with pytest.raises(FrameIngestError):
            load_frame(p)

    def test_fits_with_no_image_extension(self, tmp_path) -> None:
        """FITS with only a table HDU and no image data raises FrameIngestError."""
        col = fits.Column(name="x", format="E", array=np.array([1.0, 2.0]))
        table_hdu = fits.BinTableHDU.from_columns([col])
        primary = fits.PrimaryHDU()
        for k, v in _REQUIRED_HEADER_VALS.items():
            primary.header[k] = v
        hdul = fits.HDUList([primary, table_hdu])
        p = tmp_path / "table_only.fits"
        hdul.writeto(p, overwrite=True)
        with pytest.raises(FrameIngestError, match="No 2-D image"):
            load_frame(p)
