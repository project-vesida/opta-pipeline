"""Video-derived frame → I-01 FITS bridge.

Turns extracted image frames into I-01 FITS (with the six
required headers ``DATE-OBS, NODE-ID, EXPTIME, INSTRUME, GAIN, TEMP``) plus a
``star_matches.json`` sidecar, so the full "video in → tracklets out" path is
runnable and testable end to end.

The pipeline's input fidelity depends on preserving the original ADU bit depth:
a faint mv-13 object is sub-threshold per frame, so the extracted frames and any
intermediate video container must be **16-bit lossless** (an 8-bit lossy MP4
would destroy the signal).  Frames are therefore read as ``uint16``.  The
optional ffmpeg encode/extract helpers use FFV1-in-MKV ``gray16le``; when ffmpeg
is unavailable the 16-bit PNG sequence *is* the lossless capture.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.time import Time

__all__ = [
    "write_frames_png",
    "frames_to_fits",
    "write_star_matches",
    "ffmpeg_available",
    "ffprobe_available",
    "encode_png_sequence_to_mkv",
    "extract_mkv_to_png",
    "read_frame_uint16",
    "probe_packet_pts_s",
    "probe_container_duration_s",
    "check_pts_continuity",
    "PtsContinuityReport",
]

_IMAGE_EXTS = {".png", ".tif", ".tiff"}


def read_frame_uint16(path: str | Path) -> np.ndarray:
    """Read a single 16-bit grayscale image frame as a 2-D ``uint16`` array."""
    from PIL import Image

    with Image.open(path) as img:
        arr = np.asarray(img)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale frame, got {arr.shape}: {path}")
    if arr.dtype != np.uint16:
        # 16-bit PNGs may decode as int32 (mode 'I'); clamp and narrow.
        arr = np.clip(arr, 0, 65535).astype(np.uint16)
    return arr


def list_frame_files(frames_dir: str | Path) -> list[Path]:
    """Return sorted image-frame paths under ``frames_dir`` (non-recursive)."""
    d = Path(frames_dir)
    return sorted(p for p in d.iterdir() if p.suffix.lower() in _IMAGE_EXTS)


def write_frames_png(
    frames: Iterable[np.ndarray],
    frames_dir: str | Path,
    *,
    prefix: str = "frame_",
    start_index: int = 1,
) -> list[Path]:
    """Write ``uint16`` frames as a lossless 16-bit PNG sequence.

    This stands in for the ffmpeg frame-extraction step when no real video
    container is used; the resulting PNGs are bit-exact with the input ADU.
    """
    from PIL import Image

    out = Path(frames_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, frame in enumerate(frames):
        arr = np.asarray(frame)
        if arr.ndim != 2:
            raise ValueError(f"Frame {i} is not 2-D: shape {arr.shape}")
        path = out / f"{prefix}{start_index + i:06d}.png"
        Image.fromarray(arr.astype(np.uint16)).save(path)
        paths.append(path)
    return paths


def frames_to_fits(
    frames_dir: str | Path,
    raw_dir: str | Path,
    *,
    start_mjd: float,
    fps: float,
    node_id: str,
    instrume: str,
    gain_e_adu: float,
    temp_c: float,
    exptime_s: float | None = None,
    expected_duration_s: float | None = None,
    duration_tol_s: float | None = None,
) -> list[Path]:
    """Convert an extracted 16-bit frame sequence to I-01 FITS files.

    Each frame ``i`` (sorted by filename) is written to ``raw_dir`` with the six
    headers required by :func:`opta_pipeline.ingest.validate_header`.  The
    per-frame timestamp is ``DATE-OBS = start_mjd + i / fps`` (seconds), rendered
    as ISO 8601 UTC so :func:`opta_pipeline.ingest.date_obs_to_mjd`
    recovers it exactly.

    Parameters
    ----------
    frames_dir, raw_dir : path
        Input image-frame directory and output FITS directory (created if absent).
    start_mjd : float
        UTC Modified Julian Date of the first frame's start of exposure.
    fps : float
        Capture frame rate (Hz); sets both cadence and the default exposure.
    node_id, instrume : str
        I-01 ``NODE-ID`` and ``INSTRUME`` header values.
    gain_e_adu, temp_c : float
        I-01 ``GAIN`` (e⁻/ADU) and ``TEMP`` (°C) header values.
    exptime_s : float | None
        ``EXPTIME`` seconds; defaults to ``1 / fps`` when ``None``.
    expected_duration_s : float | None
        Wall-clock capture duration (field log / container metadata).  When
        given, ``n_frames / fps`` is cross-checked against it: the constant
        ``i / fps`` timestamp model is **dropped-frame-blind**, so a capture
        that lost frames would silently compress the FITS timeline and skew
        every downstream epoch.  A mismatch beyond ``duration_tol_s`` raises.
    duration_tol_s : float | None
        Tolerance for the duration cross-check; defaults to
        ``max(1.5 / fps, 0.005 · expected_duration_s)`` (one dropped frame is
        detectable, small container-rounding slack is not flagged).

    Returns
    -------
    list[Path]
        Written FITS paths, in frame order.
    """
    from astropy.io import fits

    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    exptime = float(exptime_s) if exptime_s is not None else 1.0 / fps
    dt_days = (1.0 / fps) / 86400.0

    frame_paths = list_frame_files(frames_dir)
    if not frame_paths:
        exts = sorted(_IMAGE_EXTS)
        raise FileNotFoundError(f"No {exts} image frames in {frames_dir}")

    if expected_duration_s is not None:
        implied_s = len(frame_paths) / fps
        tol = (
            float(duration_tol_s)
            if duration_tol_s is not None
            else max(1.5 / fps, 0.005 * float(expected_duration_s))
        )
        if abs(implied_s - float(expected_duration_s)) > tol:
            missing = round((float(expected_duration_s) - implied_s) * fps)
            raise ValueError(
                f"frame_count/fps duration cross-check failed: "
                f"{len(frame_paths)} frames at {fps:g} fps span {implied_s:.3f} s "
                f"but the capture wall clock says {float(expected_duration_s):.3f} s "
                f"(|delta| > {tol:.3f} s; ~{missing:+d} frames vs expectation). "
                "The i/fps timestamp model is dropped-frame-blind - fix the "
                "extraction (use -vsync 0 passthrough, check PTS continuity) "
                "before writing FITS timestamps."
            )

    out_dir = Path(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for i, src in enumerate(frame_paths):
        data = read_frame_uint16(src)
        date_obs = Time(start_mjd + i * dt_days, format="mjd", scale="utc").isot

        hdu = fits.PrimaryHDU(data=data)
        hdu.header["DATE-OBS"] = date_obs
        hdu.header["NODE-ID"] = node_id
        hdu.header["EXPTIME"] = exptime
        hdu.header["INSTRUME"] = instrume
        hdu.header["GAIN"] = gain_e_adu
        hdu.header["TEMP"] = temp_c

        dest = out_dir / f"{src.stem}.fits"
        hdu.writeto(dest, overwrite=True)
        written.append(dest)
    return written


def write_star_matches(path: str | Path, matches: Sequence) -> Path:
    """Write a ``star_matches.json`` sidecar from ``StarMatch``-like objects.

    Each entry must expose ``x_px, y_px, ra_deg, dec_deg`` (e.g. the output of
    :func:`opta_pipeline.synth.catalog.star_field_at`).  The format matches
    :func:`opta_pipeline.ingest.load_star_matches`.
    """
    records = [
        {
            "x_px": float(m.x_px),
            "y_px": float(m.y_px),
            "ra_deg": float(m.ra_deg),
            "dec_deg": float(m.dec_deg),
        }
        for m in matches
    ]
    out = Path(path)
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Optional ffmpeg container round-trip (16-bit lossless FFV1/gray16le in MKV)
# ---------------------------------------------------------------------------


def ffmpeg_available(ffmpeg: str = "ffmpeg") -> bool:
    """Return True if an ffmpeg executable is on PATH."""
    return shutil.which(ffmpeg) is not None


def ffprobe_available(ffprobe: str = "ffprobe") -> bool:
    """Return True if an ffprobe executable is on PATH."""
    return shutil.which(ffprobe) is not None


def probe_packet_pts_s(
    video_path: str | Path, *, ffprobe: str = "ffprobe"
) -> list[float]:
    """Video-stream packet presentation timestamps (seconds), sorted.

    Packet PTS (no decode) is the container's own record of when each captured
    frame belongs on the wall clock — the ground truth the extracted PNG
    sequence must be checked against before ``i / fps`` timestamps are trusted.
    """
    if not ffprobe_available(ffprobe):
        raise RuntimeError("ffprobe not found; cannot verify PTS continuity")
    cmd = [
        ffprobe, "-hide_banner", "-loglevel", "error",
        "-select_streams", "v:0",
        "-show_entries", "packet=pts_time",
        "-of", "csv=p=0",
        str(video_path),
    ]
    out = subprocess.run(
        cmd, check=True, capture_output=True, text=True
    ).stdout
    pts = [
        float(token)
        for line in out.splitlines()
        for token in [line.strip().rstrip(",")]
        if token and token != "N/A"
    ]
    return sorted(pts)


def probe_container_duration_s(
    video_path: str | Path, *, ffprobe: str = "ffprobe"
) -> float:
    """Container-level duration (seconds) from ffprobe format metadata."""
    if not ffprobe_available(ffprobe):
        raise RuntimeError("ffprobe not found; cannot probe container duration")
    cmd = [
        ffprobe, "-hide_banner", "-loglevel", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(video_path),
    ]
    out = subprocess.run(
        cmd, check=True, capture_output=True, text=True
    ).stdout.strip()
    return float(out)


@dataclass(frozen=True)
class PtsContinuityReport:
    """Result of a PTS-continuity check over one video stream.

    Attributes:
        n_frames: Number of PTS samples inspected.
        nominal_dt_s: Cadence the deltas were checked against (given or the
            median inter-frame delta).
        n_gaps: Count of inter-frame deltas deviating from ``nominal_dt_s``
            by more than the tolerance (dropped/duplicated/late frames).
        max_abs_deviation_s: Worst |delta − nominal| observed.
        gap_indices: Frame indices (of the *earlier* frame) where deviations
            occur, truncated to the first 20.
    """

    n_frames: int
    nominal_dt_s: float
    n_gaps: int
    max_abs_deviation_s: float
    gap_indices: tuple[int, ...]

    @property
    def ok(self) -> bool:
        """True when every inter-frame delta matched the nominal cadence."""
        return self.n_gaps == 0


def check_pts_continuity(
    pts_s: Sequence[float],
    *,
    nominal_dt_s: float | None = None,
    tol_frac: float = 0.5,
) -> PtsContinuityReport:
    """Check that PTS samples advance at a constant cadence.

    A delta deviating from the nominal cadence by more than
    ``tol_frac x nominal`` marks a dropped (delta ~ 2/fps), duplicated
    (delta ~ 0), or late frame.  ``nominal_dt_s`` defaults to the median
    delta, which is robust as long as fewer than half the frames misbehave.
    """
    deltas = [b - a for a, b in zip(pts_s, pts_s[1:])]
    if not deltas:
        return PtsContinuityReport(
            n_frames=len(pts_s),
            nominal_dt_s=float(nominal_dt_s or 0.0),
            n_gaps=0,
            max_abs_deviation_s=0.0,
            gap_indices=(),
        )
    if nominal_dt_s is None:
        nominal_dt_s = sorted(deltas)[len(deltas) // 2]
    nominal = float(nominal_dt_s)
    tol = tol_frac * nominal
    deviations = [abs(d - nominal) for d in deltas]
    gaps = [i for i, dev in enumerate(deviations) if dev > tol]
    return PtsContinuityReport(
        n_frames=len(pts_s),
        nominal_dt_s=nominal,
        n_gaps=len(gaps),
        max_abs_deviation_s=max(deviations),
        gap_indices=tuple(gaps[:20]),
    )


def encode_png_sequence_to_mkv(
    frames_dir: str | Path,
    video_path: str | Path,
    *,
    fps: float,
    prefix: str = "frame_",
    ffmpeg: str = "ffmpeg",
) -> Path:
    """Encode a 16-bit PNG sequence to a lossless FFV1/gray16le MKV (needs ffmpeg).

    Raises ``RuntimeError`` if ffmpeg is unavailable so callers can fall back to
    operating on the PNG sequence directly.
    """
    if not ffmpeg_available(ffmpeg):
        raise RuntimeError("ffmpeg not found; use the 16-bit PNG sequence directly")
    pattern = str(Path(frames_dir) / f"{prefix}%06d.png")
    out = Path(video_path)
    cmd = [
        ffmpeg, "-hide_banner", "-y",
        "-framerate", f"{fps:g}",
        "-i", pattern,
        "-c:v", "ffv1", "-pix_fmt", "gray16le",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def extract_mkv_to_png(
    video_path: str | Path,
    frames_dir: str | Path,
    *,
    prefix: str = "frame_",
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    verify_pts: bool = True,
) -> list[Path]:
    """Extract a 16-bit MKV back to a gray16le PNG sequence (needs ffmpeg).

    Extraction runs with ``-vsync 0`` (passthrough): ffmpeg emits exactly one
    PNG per stored frame and never invents or drops frames to hit a nominal
    rate — otherwise the downstream ``i / fps`` FITS timestamps are silently
    wrong after the first dropped frame.  With ``verify_pts`` (default) the
    stream's packet PTS are additionally checked for constant cadence and a
    ``ValueError`` is raised on gaps; if ffprobe is unavailable a warning is
    printed instead (the check cannot run, which is not the same as passing).
    """
    if not ffmpeg_available(ffmpeg):
        raise RuntimeError("ffmpeg not found; use the 16-bit PNG sequence directly")
    out_dir = Path(frames_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"{prefix}%06d.png")
    cmd = [
        ffmpeg, "-hide_banner", "-y",
        "-i", str(video_path),
        "-vsync", "0",
        "-pix_fmt", "gray16le",
        pattern,
    ]
    subprocess.run(cmd, check=True)
    extracted = list_frame_files(out_dir)
    if verify_pts:
        if ffprobe_available(ffprobe):
            report = check_pts_continuity(
                probe_packet_pts_s(video_path, ffprobe=ffprobe)
            )
            if not report.ok:
                raise ValueError(
                    f"PTS continuity check failed for {video_path}: "
                    f"{report.n_gaps} gap(s) over {report.n_frames} frames "
                    f"(nominal dt {report.nominal_dt_s:.6f} s, worst deviation "
                    f"{report.max_abs_deviation_s:.6f} s, first gaps at frame "
                    f"indices {list(report.gap_indices)}). The capture dropped "
                    "or duplicated frames; i/fps FITS timestamps would be wrong."
                )
        else:
            print(
                f"WARNING: ffprobe not found - PTS continuity of {video_path} "
                "NOT verified; i/fps FITS timestamps are unchecked against "
                "dropped frames."
            )
    return extracted
