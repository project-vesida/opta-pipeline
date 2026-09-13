"""Tests for the CCSDS TDM (KVN) emitter (opta_pipeline.tdm).

Structure assertions follow CCSDS 503.0-B-2 (tables 3-2/3-3, sections 3.4,
3.5.4, 4.3.9) and its ground-based optical example (figure E-16).  The
round-trip tests use a deliberately minimal test-only KVN parser to confirm
that epochs and angles survive serialisation at full float64 precision.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import pytest

from opta_pipeline.tdm import TDM_VERSION, format_tdm, mjd_to_ccsds_epoch, write_tdm
from opta_pipeline.tracklet import Tracklet, TrackletPoint

_MJD_EPOCH = datetime(1858, 11, 17, tzinfo=UTC)

# One microsecond in days — the emitter's epoch resolution (and the float64
# MJD resolution floor near the current epoch), with headroom for rounding.
_EPOCH_TOL_DAYS = 2.0e-11

_EPOCH_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}$")

CREATION = datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Test-only minimal KVN parser
# ---------------------------------------------------------------------------


def parse_kvn_tdm(
    text: str,
) -> tuple[dict[str, str], list[tuple[dict[str, str], list[tuple[str, str, float]]]]]:
    """Parse a KVN TDM into (header, [(metadata, data_records), ...]).

    Data records are (keyword, epoch_string, value) tuples.  Test-only:
    just enough KVN to verify the emitter, not a general TDM reader.
    """
    header: dict[str, str] = {}
    segments: list[tuple[dict[str, str], list[tuple[str, str, float]]]] = []
    meta: dict[str, str] | None = None
    data: list[tuple[str, str, float]] | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "META_START":
            assert meta is None and data is None
            meta = {}
            continue
        if line == "META_STOP":
            assert meta is not None
            continue
        if line == "DATA_START":
            assert meta is not None and data is None
            data = []
            continue
        if line == "DATA_STOP":
            assert meta is not None and data is not None
            segments.append((meta, data))
            meta = None
            data = None
            continue
        key, eq, value = line.partition("=")
        assert eq == "=", f"non keyword=value line: {raw!r}"
        key, value = key.strip(), value.strip()
        if data is not None:
            epoch_str, num_str = value.split()
            data.append((key, epoch_str, float(num_str)))
        elif meta is not None:
            meta[key] = value
        else:
            header[key] = value

    assert meta is None and data is None, "unterminated segment"
    return header, segments


def epoch_to_mjd(epoch: str) -> float:
    """Parse a CCSDS ASCII Time Code A epoch back to UTC MJD."""
    dt = datetime.strptime(epoch, "%Y-%m-%dT%H:%M:%S.%f").replace(
        tzinfo=UTC
    )
    return (dt - _MJD_EPOCH) / timedelta(days=1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tracklet(
    object_id: str,
    t0_mjd: float,
    ra0: float,
    dec0: float,
    *,
    n: int = 4,
    dt_s: float = 0.2,
    ra_step: float = 0.0123456789,
    dec_step: float = -0.0034567891,
) -> Tracklet:
    """Synthetic tracklet with full-precision (non-round) coordinates."""
    points = tuple(
        TrackletPoint(
            ra_deg=(ra0 + i * ra_step) % 360.0,
            dec_deg=dec0 + i * dec_step,
            utc_mjd=t0_mjd + i * dt_s / 86400.0,
            sigma_ra_arcsec=2.5,
            sigma_dec_arcsec=2.5,
        )
        for i in range(n)
    )
    return Tracklet(
        points=points,
        object_id=object_id,
        node_id="NODE-01",
        rms_arcsec=1.0,
        ra_rate_arcsec_s=ra_step * 3600.0 / dt_s,
        dec_rate_arcsec_s=dec_step * 3600.0 / dt_s,
    )


@pytest.fixture
def tracklets() -> list[Tracklet]:
    return [
        _make_tracklet("SAT-0001", 61041.5304398148, 332.2298751234567, -16.30283891),
        # Second tracklet straddles the RA 0/360 wrap.
        _make_tracklet("SAT-0002", 61041.6001157407, 359.98123456789, 42.123456789),
    ]


@pytest.fixture
def tdm_text(tracklets: list[Tracklet]) -> str:
    return format_tdm(
        tracklets,
        originator="OPTA",
        participant_1="NODE-01",
        creation_date=CREATION,
    )


# ---------------------------------------------------------------------------
# Epoch conversion
# ---------------------------------------------------------------------------


class TestEpochConversion:
    def test_known_mjd(self) -> None:
        # MJD 60000.5 = 2023-02-25T12:00:00 UTC (integer-day anchor).
        assert mjd_to_ccsds_epoch(60000.5) == "2023-02-25T12:00:00.000000"

    def test_format_shape(self) -> None:
        assert _EPOCH_RE.match(mjd_to_ccsds_epoch(61041.123456789))

    def test_round_trip_microsecond(self) -> None:
        for mjd in (60000.0, 61041.5304398148, 61041.99999998843):
            assert abs(epoch_to_mjd(mjd_to_ccsds_epoch(mjd)) - mjd) < _EPOCH_TOL_DAYS


# ---------------------------------------------------------------------------
# Message structure (CCSDS 503.0-B-2)
# ---------------------------------------------------------------------------


class TestStructure:
    def test_header(self, tdm_text: str) -> None:
        lines = tdm_text.splitlines()
        assert lines[0] == f"CCSDS_TDM_VERS = {TDM_VERSION}"
        assert lines[1] == "CREATION_DATE = 2026-07-20T12:00:00.000000"
        assert lines[2] == "ORIGINATOR = OPTA"

    def test_one_segment_per_tracklet(self, tdm_text: str) -> None:
        _, segments = parse_kvn_tdm(tdm_text)
        assert len(segments) == 2

    def test_metadata_keywords_and_order(self, tdm_text: str) -> None:
        # Fixed keyword order per table 3-3, values per figure E-16.
        meta, _ = parse_kvn_tdm(tdm_text)[1][0]
        assert list(meta) == [
            "TRACK_ID",
            "TIME_SYSTEM",
            "START_TIME",
            "STOP_TIME",
            "PARTICIPANT_1",
            "PARTICIPANT_2",
            "MODE",
            "PATH",
            "ANGLE_TYPE",
            "REFERENCE_FRAME",
        ]
        assert meta["TIME_SYSTEM"] == "UTC"
        assert meta["MODE"] == "SEQUENTIAL"
        assert meta["PATH"] == "2,1"
        assert meta["ANGLE_TYPE"] == "RADEC"
        assert meta["REFERENCE_FRAME"] == "EME2000"
        assert meta["PARTICIPANT_1"] == "NODE-01"

    def test_track_id_and_default_participant_2(self, tdm_text: str) -> None:
        _, segments = parse_kvn_tdm(tdm_text)
        assert segments[0][0]["TRACK_ID"] == "SAT-0001"
        assert segments[0][0]["PARTICIPANT_2"] == "SAT-0001"
        assert segments[1][0]["PARTICIPANT_2"] == "SAT-0002"

    def test_start_stop_span_data(self, tdm_text: str, tracklets) -> None:
        _, segments = parse_kvn_tdm(tdm_text)
        for (meta, _), trk in zip(segments, tracklets):
            assert meta["START_TIME"] == mjd_to_ccsds_epoch(trk.points[0].utc_mjd)
            assert meta["STOP_TIME"] == mjd_to_ccsds_epoch(trk.points[-1].utc_mjd)

    def test_data_records_shape(self, tdm_text: str, tracklets) -> None:
        _, segments = parse_kvn_tdm(tdm_text)
        for (_, data), trk in zip(segments, tracklets):
            # ANGLE_1/ANGLE_2 pair per point, ANGLE_1 first (figure E-16).
            assert len(data) == 2 * len(trk.points)
            assert [k for k, _, _ in data[0::2]] == ["ANGLE_1"] * len(trk.points)
            assert [k for k, _, _ in data[1::2]] == ["ANGLE_2"] * len(trk.points)
            # Paired records share the epoch.
            for a1, a2 in zip(data[0::2], data[1::2]):
                assert a1[1] == a2[1]

    def test_per_keyword_chronological_order(self, tdm_text: str) -> None:
        # Section 3.4.10: per-keyword records must be chronological.
        _, segments = parse_kvn_tdm(tdm_text)
        for _, data in segments:
            for key in ("ANGLE_1", "ANGLE_2"):
                epochs = [epoch_to_mjd(e) for k, e, _ in data if k == key]
                assert epochs == sorted(epochs)

    def test_epoch_format_everywhere(self, tdm_text: str) -> None:
        _, segments = parse_kvn_tdm(tdm_text)
        for meta, data in segments:
            assert _EPOCH_RE.match(meta["START_TIME"])
            assert _EPOCH_RE.match(meta["STOP_TIME"])
            for _, epoch, _ in data:
                assert _EPOCH_RE.match(epoch)

    def test_angle_range(self, tdm_text: str) -> None:
        # Section 3.5.4: -180.0 <= angle < 360.0 (degrees).
        _, segments = parse_kvn_tdm(tdm_text)
        for _, data in segments:
            for _, _, value in data:
                assert -180.0 <= value < 360.0

    def test_ascii_only(self, tdm_text: str) -> None:
        tdm_text.encode("ascii")


# ---------------------------------------------------------------------------
# Round-trip precision
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_angles_bit_exact(self, tdm_text: str, tracklets) -> None:
        _, segments = parse_kvn_tdm(tdm_text)
        for (_, data), trk in zip(segments, tracklets):
            ra = [v for k, _, v in data if k == "ANGLE_1"]
            dec = [v for k, _, v in data if k == "ANGLE_2"]
            assert ra == [p.ra_deg for p in trk.points]
            assert dec == [p.dec_deg for p in trk.points]

    def test_epochs_to_microsecond(self, tdm_text: str, tracklets) -> None:
        _, segments = parse_kvn_tdm(tdm_text)
        for (_, data), trk in zip(segments, tracklets):
            epochs = [epoch_to_mjd(e) for k, e, _ in data if k == "ANGLE_1"]
            for parsed, p in zip(epochs, trk.points):
                assert abs(parsed - p.utc_mjd) < _EPOCH_TOL_DAYS


# ---------------------------------------------------------------------------
# API behaviour
# ---------------------------------------------------------------------------


class TestSmallAngleFormatting:
    """|angle| < 1e-4 deg must stay positional — KVN admits no exponent.

    ``repr(3e-05)`` is ``'3e-05'``; a declination of a few tenths of a
    milliarcsecond (or an RA that lands just past the 0/360 wrap) is a legal
    angle, and the exponent form is not a KVN value.  The shortest-round-trip
    contract (``test_angles_bit_exact``) must survive the change.
    """

    _SMALL = [3e-05, -3e-05, 1e-07, 9.87654321e-06, 1.5e-300]

    @pytest.mark.parametrize("angle", _SMALL)
    def test_small_angles_have_no_exponent(self, angle: float) -> None:
        trk = Tracklet(
            points=(
                TrackletPoint(
                    ra_deg=angle,
                    dec_deg=angle,
                    utc_mjd=61041.5,
                    sigma_ra_arcsec=2.5,
                    sigma_dec_arcsec=2.5,
                ),
            ),
            object_id="SAT-TINY",
            node_id="NODE-01",
            rms_arcsec=1.0,
            ra_rate_arcsec_s=0.0,
            dec_rate_arcsec_s=0.0,
        )
        text = format_tdm(
            [trk],
            originator="OPTA",
            participant_1="NODE-01",
            creation_date=CREATION,
        )
        angle_lines = [
            ln
            for ln in text.splitlines()
            if ln.startswith(("ANGLE_1 =", "ANGLE_2 ="))
        ]
        assert angle_lines
        for line in angle_lines:
            value = line.split("=", 1)[1].split()[-1]
            assert "e" not in value and "E" not in value, line

        _, segments = parse_kvn_tdm(text)
        _, data = segments[0]
        assert [v for _, _, v in data] == [angle, angle]


class TestApi:
    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one tracklet"):
            format_tdm([], originator="OPTA", participant_1="NODE-01")

    def test_tracklet_without_points_raises(self) -> None:
        """A zero-point tracklet used to IndexError on points[0] (START_TIME)."""
        empty = Tracklet(
            points=(),
            object_id="SAT-EMPTY",
            node_id="NODE-01",
            rms_arcsec=1.0,
            ra_rate_arcsec_s=0.0,
            dec_rate_arcsec_s=0.0,
        )
        with pytest.raises(ValueError, match="has no points"):
            format_tdm([empty], originator="OPTA", participant_1="NODE-01")

    def test_participant_2_resolver(self, tracklets) -> None:
        names = {"SAT-0001": "ISS (ZARYA)", "SAT-0002": "1998-067A"}
        text = format_tdm(
            tracklets,
            originator="OPTA",
            participant_1="NODE-01",
            participant_2_resolver=lambda t: names[t.object_id],
            creation_date=CREATION,
        )
        _, segments = parse_kvn_tdm(text)
        assert segments[0][0]["PARTICIPANT_2"] == "ISS (ZARYA)"
        assert segments[1][0]["PARTICIPANT_2"] == "1998-067A"

    def test_reference_frame_override(self, tracklets) -> None:
        text = format_tdm(
            tracklets,
            originator="OPTA",
            participant_1="NODE-01",
            reference_frame="ICRF",
            creation_date=CREATION,
        )
        _, segments = parse_kvn_tdm(text)
        assert all(m["REFERENCE_FRAME"] == "ICRF" for m, _ in segments)

    def test_naive_creation_date_taken_as_utc(self, tracklets) -> None:
        text = format_tdm(
            tracklets,
            originator="OPTA",
            participant_1="NODE-01",
            creation_date=datetime(2026, 7, 20, 12, 0, 0),
        )
        assert "CREATION_DATE = 2026-07-20T12:00:00.000000" in text

    def test_write_tdm(self, tracklets, tmp_path) -> None:
        out = write_tdm(
            tracklets,
            tmp_path / "sub" / "tracklets.tdm",
            originator="OPTA",
            participant_1="NODE-01",
            creation_date=CREATION,
        )
        assert out == tmp_path / "sub" / "tracklets.tdm"
        assert out.read_text(encoding="ascii") == format_tdm(
            tracklets,
            originator="OPTA",
            participant_1="NODE-01",
            creation_date=CREATION,
        )
