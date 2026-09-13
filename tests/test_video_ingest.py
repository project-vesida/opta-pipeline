"""Tests for the video-derived frame -> I-01 FITS bridge.

The bridge's whole reason to exist is fidelity: a faint mv-13 object is
sub-threshold per frame, so the capture must survive ingest **bit-exactly**
(an 8-bit lossy container would destroy it).  These tests assert the uint16
ADU round-trips through the PNG sequence, the optional ffmpeg/FFV1 container,
and the FITS bridge, and that the emitted FITS satisfy the I-01 header contract.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from opta_pipeline.astrometry import StarMatch
from opta_pipeline.ingest import (
    _REQUIRED_HEADERS,
    adu_to_electrons,
    date_obs_to_mjd,
    load_frame,
    load_star_matches,
    validate_header,
)
from opta_pipeline.ingest_video import (
    check_pts_continuity,
    encode_png_sequence_to_mkv,
    extract_mkv_to_png,
    ffmpeg_available,
    ffprobe_available,
    frames_to_fits,
    probe_packet_pts_s,
    read_frame_uint16,
    write_frames_png,
    write_star_matches,
)

_FPS = 25.0
_GAIN = 1.7
_NODE = "NODE-VID"
_INSTRUME = "IMX585-TEST"
_TEMP = -5.0
_MJD0 = 60000.0


def _synthetic_frames(n: int = 4, *, w: int = 16, h: int = 12) -> list[np.ndarray]:
    """Deterministic uint16 frames spanning the full 16-bit range."""
    rng = np.random.default_rng(1234)
    frames = []
    for i in range(n):
        arr = rng.integers(0, 65536, size=(h, w), dtype=np.uint16)
        arr[0, 0] = 0
        arr[-1, -1] = 65535  # exercise both extremes
        arr[1, 1] = np.uint16(1000 + i)  # frame-distinguishing pixel
        frames.append(arr)
    return frames


def test_write_frames_png_roundtrips_uint16(tmp_path) -> None:
    frames = _synthetic_frames()
    paths = write_frames_png(frames, tmp_path / "frames")

    assert len(paths) == len(frames)
    for original, path in zip(frames, paths, strict=True):
        back = read_frame_uint16(path)
        assert back.dtype == np.uint16
        np.testing.assert_array_equal(back, original)


def test_frames_to_fits_satisfies_header_contract(tmp_path) -> None:
    frames = _synthetic_frames()
    write_frames_png(frames, tmp_path / "frames")

    fits_paths = frames_to_fits(
        tmp_path / "frames",
        tmp_path / "raw",
        start_mjd=_MJD0,
        fps=_FPS,
        node_id=_NODE,
        instrume=_INSTRUME,
        gain_e_adu=_GAIN,
        temp_c=_TEMP,
    )

    assert len(fits_paths) == len(frames)
    for path in fits_paths:
        header = fits.getheader(path)
        validate_header(header)  # raises if any required keyword missing
        assert all(k in header for k in _REQUIRED_HEADERS)


def test_frames_to_fits_pixels_and_cadence(tmp_path) -> None:
    frames = _synthetic_frames()
    write_frames_png(frames, tmp_path / "frames")

    fits_paths = frames_to_fits(
        tmp_path / "frames",
        tmp_path / "raw",
        start_mjd=_MJD0,
        fps=_FPS,
        node_id=_NODE,
        instrume=_INSTRUME,
        gain_e_adu=_GAIN,
        temp_c=_TEMP,
    )

    dt_days = (1.0 / _FPS) / 86400.0
    for i, path in enumerate(fits_paths):
        header, adu = load_frame(path)
        # Pixel data preserved bit-exactly through the FITS bridge.
        np.testing.assert_array_equal(adu, frames[i])
        # ADU -> electrons uses the header gain.
        np.testing.assert_allclose(
            adu_to_electrons(adu, header.gain), frames[i] * _GAIN
        )
        # DATE-OBS encodes the per-frame cadence and recovers exactly.
        mjd = date_obs_to_mjd(header.date_obs)
        assert mjd == pytest.approx(_MJD0 + i * dt_days, abs=1e-9)
        assert header.exptime_s == pytest.approx(1.0 / _FPS)
        assert header.node_id == _NODE
        assert header.instrume == _INSTRUME
        assert header.gain == pytest.approx(_GAIN)
        assert header.temp_c == pytest.approx(_TEMP)


def test_frames_to_fits_rejects_nonpositive_fps(tmp_path) -> None:
    write_frames_png(_synthetic_frames(), tmp_path / "frames")
    with pytest.raises(ValueError, match="fps must be positive"):
        frames_to_fits(
            tmp_path / "frames",
            tmp_path / "raw",
            start_mjd=_MJD0,
            fps=0.0,
            node_id=_NODE,
            instrume=_INSTRUME,
            gain_e_adu=_GAIN,
            temp_c=_TEMP,
        )


def test_frames_to_fits_empty_dir_raises(tmp_path) -> None:
    (tmp_path / "frames").mkdir()
    with pytest.raises(FileNotFoundError):
        frames_to_fits(
            tmp_path / "frames",
            tmp_path / "raw",
            start_mjd=_MJD0,
            fps=_FPS,
            node_id=_NODE,
            instrume=_INSTRUME,
            gain_e_adu=_GAIN,
            temp_c=_TEMP,
        )


def test_star_matches_roundtrip(tmp_path) -> None:
    matches = [
        StarMatch(x_px=10.5, y_px=20.25, ra_deg=135.0, dec_deg=45.0),
        StarMatch(x_px=200.0, y_px=150.0, ra_deg=135.1, dec_deg=44.9),
    ]
    path = write_star_matches(tmp_path / "star_matches.json", matches)
    loaded = load_star_matches(path)

    assert len(loaded) == len(matches)
    for orig, got in zip(matches, loaded, strict=True):
        assert got.x_px == pytest.approx(orig.x_px)
        assert got.y_px == pytest.approx(orig.y_px)
        assert got.ra_deg == pytest.approx(orig.ra_deg)
        assert got.dec_deg == pytest.approx(orig.dec_deg)


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not on PATH")
def test_ffv1_container_roundtrip_is_lossless(tmp_path) -> None:
    """ndarray -> FFV1/gray16le MKV -> PNG must preserve uint16 exactly."""
    frames = _synthetic_frames(n=3)
    write_frames_png(frames, tmp_path / "frames")

    video = encode_png_sequence_to_mkv(
        tmp_path / "frames", tmp_path / "capture.mkv", fps=_FPS
    )
    extracted = extract_mkv_to_png(video, tmp_path / "extracted")

    assert len(extracted) == len(frames)
    for original, path in zip(frames, extracted, strict=True):
        np.testing.assert_array_equal(read_frame_uint16(path), original)


# ---------------------------------------------------------------------------
# Pre-capture hygiene: dropped-frame detection on the extract side
# ---------------------------------------------------------------------------


def test_check_pts_continuity_clean_cadence() -> None:
    pts = [i / _FPS for i in range(50)]
    report = check_pts_continuity(pts)
    assert report.ok
    assert report.n_frames == 50
    assert report.n_gaps == 0
    assert report.nominal_dt_s == pytest.approx(1.0 / _FPS)


def test_check_pts_continuity_detects_dropped_frame() -> None:
    pts = [i / _FPS for i in range(50)]
    del pts[20]  # one dropped frame -> a 2/fps gap between indices 19 and 20
    report = check_pts_continuity(pts)
    assert not report.ok
    assert report.n_gaps == 1
    assert report.gap_indices == (19,)
    assert report.max_abs_deviation_s == pytest.approx(1.0 / _FPS)


def test_check_pts_continuity_detects_duplicated_frame() -> None:
    pts = sorted([i / _FPS for i in range(50)] + [10 / _FPS])
    report = check_pts_continuity(pts)
    assert not report.ok
    assert report.n_gaps >= 1


def test_check_pts_continuity_explicit_nominal() -> None:
    # Every delta is 2x the declared cadence -> all flagged.
    pts = [i * 2.0 / _FPS for i in range(10)]
    report = check_pts_continuity(pts, nominal_dt_s=1.0 / _FPS)
    assert report.n_gaps == 9


def test_check_pts_continuity_degenerate_inputs() -> None:
    assert check_pts_continuity([]).ok
    assert check_pts_continuity([0.0]).ok


def test_frames_to_fits_duration_crosscheck_passes(tmp_path) -> None:
    frames = _synthetic_frames()  # 4 frames
    write_frames_png(frames, tmp_path / "frames")
    paths = frames_to_fits(
        tmp_path / "frames",
        tmp_path / "raw",
        start_mjd=_MJD0,
        fps=_FPS,
        node_id=_NODE,
        instrume=_INSTRUME,
        gain_e_adu=_GAIN,
        temp_c=_TEMP,
        expected_duration_s=len(frames) / _FPS,
    )
    assert len(paths) == len(frames)


def test_frames_to_fits_duration_crosscheck_catches_dropped_frames(
    tmp_path,
) -> None:
    """A wall clock 2 frame-periods longer than n/fps means frames were lost."""
    frames = _synthetic_frames()  # 4 frames = 0.16 s at 25 fps
    write_frames_png(frames, tmp_path / "frames")
    with pytest.raises(ValueError, match="cross-check failed"):
        frames_to_fits(
            tmp_path / "frames",
            tmp_path / "raw",
            start_mjd=_MJD0,
            fps=_FPS,
            node_id=_NODE,
            instrume=_INSTRUME,
            gain_e_adu=_GAIN,
            temp_c=_TEMP,
            expected_duration_s=(len(frames) + 2) / _FPS,
        )


@pytest.mark.skipif(
    not (ffmpeg_available() and ffprobe_available()),
    reason="ffmpeg/ffprobe not on PATH",
)
def test_extract_verifies_pts_continuity_on_clean_container(tmp_path) -> None:
    """A cleanly encoded container passes the extract-side PTS verification."""
    frames = _synthetic_frames(n=6)
    write_frames_png(frames, tmp_path / "frames")
    video = encode_png_sequence_to_mkv(
        tmp_path / "frames", tmp_path / "capture.mkv", fps=_FPS
    )
    extracted = extract_mkv_to_png(video, tmp_path / "extracted", verify_pts=True)
    assert len(extracted) == len(frames)
    pts = probe_packet_pts_s(video)
    assert len(pts) == len(frames)
    assert check_pts_continuity(pts, nominal_dt_s=1.0 / _FPS).ok
