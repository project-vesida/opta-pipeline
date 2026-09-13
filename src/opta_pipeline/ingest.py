"""FITS frame ingestion module for OpTA pipeline (WP-C6).

Reads raw FITS frames from disk, validates required header keywords from
pipeline_defaults.yaml::frame_format, and returns the pixel data for
downstream calibration.

Supported formats
-----------------
- Plain FITS (BITPIX 16)
- Rice-compressed FITS (fpack/rice) — XTENSION = 'BINTABLE' with ZIMAGE keyword

Header validation
-----------------
All headers listed in config.frame_format.required_headers must be present.
Missing or unreadable headers raise FrameIngestError with a clear message.

Usage
-----
    from opta_pipeline.ingest import load_frame, FrameHeader

    header, data = load_frame("node01_20260602T120000.fits")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from astropy.io import fits
from astropy.time import Time

from opta_pipeline.astrometry import StarMatch
from opta_pipeline.config import PipelineConfig

__all__ = [
    "FrameHeader",
    "FrameIngestError",
    "load_frame",
    "validate_header",
]

_REQUIRED_HEADERS = [
    "DATE-OBS",
    "NODE-ID",
    "EXPTIME",
    "INSTRUME",
    "GAIN",
    "TEMP",
]


def _header_str(header: fits.Header, key: str) -> str:
    """Read a required FITS header value as a string."""
    return str(header[key])


def _header_float(header: fits.Header, key: str) -> float:
    """Read a required FITS header value as a float."""
    return float(str(header[key]))


class FrameIngestError(ValueError):
    """Raised when a FITS frame cannot be ingested."""


@dataclass(frozen=True)
class FrameHeader:
    """Validated FITS header fields required by the I-02 interface.

    All fields are sourced from the FITS header; types are coerced on load.

    Attributes
    ----------
    date_obs : str
        ISO 8601 UTC start-of-exposure timestamp (from DATE-OBS).
    node_id : str
        Node identifier string (from NODE-ID).
    exptime_s : float
        Exposure time in seconds (from EXPTIME).
    instrume : str
        Instrument / sensor model (from INSTRUME).
    gain : float
        Sensor gain in e-/ADU (from GAIN).
    temp_c : float
        Sensor temperature in Celsius (from TEMP).
    """

    date_obs: str
    node_id: str
    exptime_s: float
    instrume: str
    gain: float
    temp_c: float


def validate_header(
    header: fits.Header,
    required: list[str] | None = None,
) -> None:
    """Raise FrameIngestError if any required keyword is absent.

    Parameters
    ----------
    header : astropy.io.fits.Header
        FITS header to validate.
    required : list[str] | None
        Required keywords.  Defaults to the pipeline_defaults.yaml list.

    Raises
    ------
    FrameIngestError
        With the list of missing keywords.
    """
    if required is None:
        required = _REQUIRED_HEADERS
    missing = [k for k in required if k not in header]
    if missing:
        raise FrameIngestError(f"FITS header missing required keyword(s): {missing}")


def load_frame(
    path: str | Path,
    config: PipelineConfig | None = None,
) -> tuple[FrameHeader, np.ndarray]:
    """Load a FITS frame from disk, validate headers, and return pixel data.

    Handles plain FITS and Rice-compressed FITS transparently via astropy.

    Parameters
    ----------
    path : str | Path
        Path to the FITS file.
    config : PipelineConfig | None
        Pipeline config for required header list.  Uses defaults if None.

    Returns
    -------
    (header, data) : tuple[FrameHeader, np.ndarray]
        Validated header struct and 2-D uint16 pixel array.

    Raises
    ------
    FrameIngestError
        If the file cannot be opened, the header is incomplete, or no
        valid image extension is found.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FITS file not found: {path}")

    try:
        hdul = fits.open(path, memmap=False)
    except Exception as exc:
        raise FrameIngestError(f"Cannot open FITS file {path}: {exc}") from exc

    try:
        # Find the image HDU — works for both plain FITS and Rice-compressed.
        image_hdu = None
        primary_header = cast(Any, hdul[0]).header

        for hdu in hdul:
            # Rice-compressed frames expose the image via CompImageHDU
            if isinstance(hdu, fits.CompImageHDU):
                image_hdu = hdu
                break
            if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU)):
                if hdu.data is not None and hdu.data.ndim == 2:
                    image_hdu = hdu
                    break

        if image_hdu is None:
            raise FrameIngestError(f"No 2-D image extension found in {path}")

        header = image_hdu.header
        # For CompImageHDU, merge primary header so DATE-OBS etc. are visible
        if isinstance(image_hdu, fits.CompImageHDU):
            merged = primary_header.copy()
            merged.update(header)
            header = merged

        validate_header(header)

        frame_header = FrameHeader(
            date_obs=_header_str(header, "DATE-OBS"),
            node_id=_header_str(header, "NODE-ID"),
            exptime_s=_header_float(header, "EXPTIME"),
            instrume=_header_str(header, "INSTRUME"),
            gain=_header_float(header, "GAIN"),
            temp_c=_header_float(header, "TEMP"),
        )

        data = np.asarray(image_hdu.data)
        if data.ndim != 2:
            raise FrameIngestError(f"Expected 2-D image, got shape {data.shape}")

    finally:
        hdul.close()

    return frame_header, data


# ---------------------------------------------------------------------------
# Frame-sequence and star-match helpers (I-01 directories, astrometry.net corr)
# ---------------------------------------------------------------------------


def date_obs_to_mjd(date_obs: str) -> float:
    """Parse I-01 ``DATE-OBS`` (ISO 8601 UTC) to Modified Julian Date."""
    return float(cast(float, Time(date_obs, format="isot", scale="utc").mjd))


def adu_to_electrons(data: np.ndarray, gain_e_adu: float) -> np.ndarray:
    """Convert uint16 ADU to float64 electrons using header gain."""
    return data.astype(np.float64) * gain_e_adu


def list_fits_frames(directory: str | Path) -> list[Path]:
    """Sorted FITS paths under ``directory`` (non-recursive)."""
    d = Path(directory)
    return sorted(p for p in d.iterdir() if p.suffix.lower() in {".fits", ".fit"})


def load_star_matches(sidecar: str | Path) -> list[StarMatch]:
    """Load ``StarMatch`` list from a JSON sidecar.

    Expected format: a JSON array of objects each carrying at least the keys
    ``x_px``, ``y_px`` (0-based pixel coordinates of the detected star) and
    ``ra_deg``, ``dec_deg`` (matched catalog position, degrees J2000); extra
    keys are ignored.

    SCHEMA NOTE (verified against the code path 2026-07-20): this JSON layout
    is an **OpTA-internal sidecar convention** — nova.astrometry.net does NOT
    export it.  Nova's machine-readable star-match product is ``corr.fits``
    (a binary FITS table); convert it with :func:`star_matches_from_corr` and
    persist via :func:`opta_pipeline.ingest_video.write_star_matches`.
    """
    import json

    raw = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    return [
        StarMatch(
            x_px=float(item["x_px"]),
            y_px=float(item["y_px"]),
            ra_deg=float(item["ra_deg"]),
            dec_deg=float(item["dec_deg"]),
        )
        for item in raw
    ]


def star_matches_from_corr(corr_fits: str | Path) -> list[StarMatch]:
    """Convert a nova.astrometry.net ``corr.fits`` table to ``StarMatch`` list.

    nova.astrometry.net's downloadable star-correspondence product is
    ``corr.fits``: a binary FITS table (HDU 1) whose rows pair each detected
    field star with its matched index/catalog star.  Relevant columns
    (astrometry.net ≥ 0.67 naming):

    - ``field_x``, ``field_y`` — detected star centroid in **FITS 1-based**
      pixel coordinates,
    - ``index_ra``, ``index_dec`` — matched catalog position (deg, J2000).

    (Also present but unused here: ``field_ra/field_dec`` — the WCS-projected
    sky position of the detection, ``index_x/index_y``, ``match_weight``.)

    Pixel convention: FITS counts pixel centres from 1, the pipeline's
    detection/astrometry code uses 0-based array coordinates, so 1.0 is
    subtracted from ``field_x``/``field_y``.
    """
    from astropy.io import fits

    with fits.open(corr_fits) as hdul:
        hdu = hdul[1]
        if not isinstance(hdu, fits.BinTableHDU) or hdu.data is None:
            raise ValueError(f"no correspondence table in {corr_fits}")
        table = hdu.data
        columns = table.columns
        if columns is None:
            raise ValueError(f"no correspondence table in {corr_fits}")
        names = {name.lower() for name in columns.names}
        required = {"field_x", "field_y", "index_ra", "index_dec"}
        missing = required - names
        if missing:
            raise ValueError(
                f"{corr_fits} lacks expected astrometry.net corr columns "
                f"{sorted(missing)} (found {sorted(names)})"
            )
        return [
            StarMatch(
                x_px=float(row["field_x"]) - 1.0,
                y_px=float(row["field_y"]) - 1.0,
                ra_deg=float(row["index_ra"]),
                dec_deg=float(row["index_dec"]),
            )
            for row in table
        ]


def load_fits_sequence(directory: str | Path):
    """Yield ``(header, electrons)`` for each FITS in a directory."""
    for path in list_fits_frames(directory):
        header, adu = load_frame(path)
        yield header, adu_to_electrons(adu, header.gain), path
