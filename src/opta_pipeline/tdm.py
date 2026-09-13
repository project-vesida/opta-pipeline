"""CCSDS Tracking Data Message (TDM) emitter for tracklets.

Serialises :class:`opta_pipeline.tracklet.Tracklet` objects to a CCSDS
Tracking Data Message in Keyword = Value Notation (KVN), per CCSDS 503.0-B-2
(Blue Book, June 2020, incl. Technical Corrigendum 1, October 2021).  TDM is
the interagency exchange standard for angles-only optical tracking data —
the natural downstream format for OpTA tracklets (I-02 records are the
internal interface; TDM is the external one).

Structure (one message per file)
--------------------------------
* Header: ``CCSDS_TDM_VERS = 2.0``, ``CREATION_DATE``, ``ORIGINATOR``
  (table 3-2 of the standard; mandatory keywords only, fixed order).
* One segment per tracklet: a metadata section (table 3-3, keywords in the
  standard's fixed order) followed by a data section of
  ``ANGLE_1 = <epoch> <ra_deg>`` / ``ANGLE_2 = <epoch> <dec_deg>`` records
  (section 3.5.4: angles are degrees; ``ANGLE_TYPE = RADEC`` requires an
  inertial ``REFERENCE_FRAME``).

Modelled on the standard's own ground-based optical example (figure E-16):
``PARTICIPANT_1`` is the sensor/station, ``PARTICIPANT_2`` the tracked
object, ``MODE = SEQUENTIAL`` with ``PATH = 2,1`` (light travels from the
target, participant 2, to the sensor, participant 1).

The pipeline's plate solves are fit against J2000 catalog positions
(``opta_pipeline.astrometry`` / ``synth.catalog``), so the emitted
``REFERENCE_FRAME`` defaults to ``EME2000`` (the standard's designation for
the J2000 Earth mean equator and equinox frame).

Epochs are CCSDS ASCII Time Code A (``YYYY-MM-DDThh:mm:ss.ffffff``,
section 4.3.9), converted from the tracklet points' UTC MJD timestamps at
microsecond resolution — the resolution limit of a float64 MJD near the
current epoch is ~1 microsecond, so nothing further survives anyway.

Pure serialisation: no I/O beyond writing the one output file in
:func:`write_tdm`; :func:`format_tdm` does none at all.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from opta_pipeline.tracklet import Tracklet

__all__ = [
    "TDM_VERSION",
    "format_tdm",
    "mjd_to_ccsds_epoch",
    "write_tdm",
]

TDM_VERSION = "2.0"

# MJD epoch: 1858-11-17T00:00:00 UTC.
_MJD_EPOCH = datetime(1858, 11, 17, tzinfo=UTC)


def mjd_to_ccsds_epoch(utc_mjd: float) -> str:
    """Convert a UTC MJD to a CCSDS ASCII Time Code A string.

    Format ``YYYY-MM-DDThh:mm:ss.ffffff`` (CCSDS 503.0-B-2 section 4.3.9,
    calendar-date variant, no trailing ``Z``), with microsecond fractional
    seconds — the precision floor of a float64 MJD at the current epoch
    (2**-52 * 60000 days ≈ 1.2 us).

    Note: like practically all MJD arithmetic, this maps MJD to calendar
    time by fixed 86400 s days (leap seconds are a timescale concern that
    lives upstream, in whatever produced the UTC MJD).
    """
    dt = _MJD_EPOCH + timedelta(days=utc_mjd)
    return f"{dt:%Y-%m-%dT%H:%M:%S}.{dt.microsecond:06d}"


def _format_angle(value_deg: float) -> str:
    """Shortest positional decimal that round-trips the float64 angle exactly.

    ``repr`` switches to exponent notation below 1e-4 (``3e-05``), which the
    KVN value grammar (503.0-B-2 section 6.4) does not admit — a declination
    of a few tenths of a milliarcsecond is a legal angle and must not emit an
    ``e``.  ``unique=True`` keeps the shortest round-tripping digit string, so
    the bit-exactness contract is unchanged; ``trim='0'`` keeps one digit past
    the point (``180.0``, not ``180``) so every angle stays visibly real-valued
    as ``repr`` had it.
    """
    return np.format_float_positional(float(value_deg), unique=True, trim="0")


def _segment_lines(
    tracklet: Tracklet,
    *,
    participant_1: str,
    participant_2: str,
    reference_frame: str,
) -> list[str]:
    """One TDM segment (metadata + data sections) for a tracklet.

    Metadata keywords follow the fixed order of table 3-3; the data section
    emits the ``ANGLE_1``/``ANGLE_2`` pair per point in chronological order
    (section 3.4.10 requires per-keyword chronological order; pairing the
    two angles per epoch matches figure E-16).

    A tracklet with no points has no START_TIME/STOP_TIME and would emit an
    empty DATA section, so it is rejected here — same family as
    :func:`format_tdm`'s empty-tracklets check.
    """
    points = sorted(tracklet.points, key=lambda p: p.utc_mjd)
    if not points:
        raise ValueError(
            f"tracklet {tracklet.object_id!r} has no points; a TDM segment "
            "requires at least one ANGLE_1/ANGLE_2 pair"
        )
    lines = [
        "META_START",
        f"TRACK_ID = {tracklet.object_id}",
        "TIME_SYSTEM = UTC",
        f"START_TIME = {mjd_to_ccsds_epoch(points[0].utc_mjd)}",
        f"STOP_TIME = {mjd_to_ccsds_epoch(points[-1].utc_mjd)}",
        f"PARTICIPANT_1 = {participant_1}",
        f"PARTICIPANT_2 = {participant_2}",
        "MODE = SEQUENTIAL",
        "PATH = 2,1",
        "ANGLE_TYPE = RADEC",
        f"REFERENCE_FRAME = {reference_frame}",
        "META_STOP",
        "DATA_START",
    ]
    for p in points:
        epoch = mjd_to_ccsds_epoch(p.utc_mjd)
        lines.append(f"ANGLE_1 = {epoch} {_format_angle(p.ra_deg)}")
        lines.append(f"ANGLE_2 = {epoch} {_format_angle(p.dec_deg)}")
    lines.append("DATA_STOP")
    return lines


def format_tdm(
    tracklets: Sequence[Tracklet],
    *,
    originator: str,
    participant_1: str,
    participant_2_resolver: Callable[[Tracklet], str] | None = None,
    reference_frame: str = "EME2000",
    creation_date: datetime | None = None,
) -> str:
    """Serialise tracklets to a CCSDS TDM (KVN) string.

    Parameters
    ----------
    tracklets : Sequence[Tracklet]
        Tracklets to emit, one TDM segment each.  Must be non-empty (a TDM
        requires at least one segment).
    originator : str
        ``ORIGINATOR`` header value — the creating agency/institution.
    participant_1 : str
        Sensor/station identifier (``PARTICIPANT_1``), e.g. the I-02
        ``node_id``.
    participant_2_resolver : Callable[[Tracklet], str], optional
        Maps a tracklet to its ``PARTICIPANT_2`` (target) identifier — the
        associated satellite name or NORAD/COSPAR id when a correlation is
        known.  Default: the tracklet's ``object_id`` (an uncorrelated
        detection has no better name; the standard explicitly allows
        placeholder participants for initial space-surveillance detections).
    reference_frame : str
        ``REFERENCE_FRAME`` for the RADEC angles.  Default ``EME2000``
        (J2000) — what the plate-solve path actually produces.
    creation_date : datetime, optional
        Header ``CREATION_DATE`` (UTC).  Defaults to now; pass a fixed value
        for reproducible output.  Naive datetimes are taken as UTC.

    Returns
    -------
    str
        The complete KVN message, newline-terminated.
    """
    if not tracklets:
        raise ValueError("TDM requires at least one tracklet (one segment)")

    if creation_date is None:
        creation_date = datetime.now(UTC)
    elif creation_date.tzinfo is not None:
        creation_date = creation_date.astimezone(UTC)

    lines = [
        f"CCSDS_TDM_VERS = {TDM_VERSION}",
        f"CREATION_DATE = {creation_date:%Y-%m-%dT%H:%M:%S}"
        f".{creation_date.microsecond:06d}",
        f"ORIGINATOR = {originator}",
    ]
    for tracklet in tracklets:
        participant_2 = (
            participant_2_resolver(tracklet)
            if participant_2_resolver is not None
            else tracklet.object_id
        )
        lines.append("")  # blank separator, as in the standard's examples
        lines.extend(
            _segment_lines(
                tracklet,
                participant_1=participant_1,
                participant_2=participant_2,
                reference_frame=reference_frame,
            )
        )
    return "\n".join(lines) + "\n"


def write_tdm(
    tracklets: Sequence[Tracklet],
    path: str | Path,
    *,
    originator: str,
    participant_1: str,
    participant_2_resolver: Callable[[Tracklet], str] | None = None,
    reference_frame: str = "EME2000",
    creation_date: datetime | None = None,
) -> Path:
    """Write tracklets to a CCSDS TDM (KVN) file; return the written path.

    Thin file wrapper over :func:`format_tdm` (see there for parameters).
    Parent directories are created as needed.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        format_tdm(
            tracklets,
            originator=originator,
            participant_1=participant_1,
            participant_2_resolver=participant_2_resolver,
            reference_frame=reference_frame,
            creation_date=creation_date,
        ),
        encoding="ascii",
    )
    return out
