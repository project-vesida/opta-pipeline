# Interfaces

The pipeline sits between a camera and the Vesida network. Three contracts define that seat. Their
identifiers (I-01, I-02) come from the OpTA systems engineering in
[opta-engineering/SYSTEMS.md](https://github.com/project-vesida/opta-engineering/blob/main/SYSTEMS.md).

## I-01: sensor frames in

One FITS file per frame, 16-bit unsigned pixels, with these header keywords. `opta_pipeline.ingest.validate_header`
is the normative check; `load_frame` applies it.

| Keyword | Meaning |
|---|---|
| `DATE-OBS` | Frame mid-exposure time, ISO 8601 UTC (`YYYY-MM-DDThh:mm:ss.ffffff`) |
| `NODE-ID` | Station identifier, carried through to every tracklet record |
| `EXPTIME` | Exposure time, seconds |
| `INSTRUME` | Camera / sensor identifier |
| `GAIN` | Electrons per ADU, used to convert pixel values to electrons |
| `TEMP` | Sensor temperature, degrees Celsius |

Timing is the contract's hard part. The pipeline treats `DATE-OBS` as truth; the node is responsible for
stamping it from a GNSS-disciplined clock (interface I-05 in SYSTEMS.md). Rolling-shutter readout is
corrected inside the pipeline from `rolling_shutter.row_readout_us` in the config.

`opta_pipeline.ingest_video` turns a camera video or a PNG sequence into I-01 FITS for cameras that
cannot write FITS directly. It needs `ffmpeg` and `ffprobe` on the PATH.

## I-02: tracklet records out

A tracklet is a time-ordered set of astrometric positions of one moving object. `tracklets_to_records`
flattens tracklets into JSON records, one per observation:

```json
{"ra": 135.02, "dec": 45.01, "utc": 60000.0417,
 "sigma_ra": 3.2, "sigma_dec": 3.1,
 "object_id": "TRK-0001", "node_id": "NODE-01"}
```

`ra`, `dec` are degrees in the J2000 frame (the plate solve fits against J2000 catalog positions).
`utc` is Modified Julian Date. `sigma_ra`, `sigma_dec` are arcseconds. `object_id` is unique within a run;
tracklets from the blind track-and-stack path carry the `STK-` prefix. `node_id` is the I-01 `NODE-ID`.

## CCSDS Tracking Data Message

`opta_pipeline.tdm` writes the same tracklets as a CCSDS TDM (503.0-B-2, KVN), the interagency exchange
format for angles-only optical observations. One segment per tracklet, `ANGLE_TYPE = RADEC`,
`REFERENCE_FRAME = EME2000`, `PARTICIPANT_1` the station, `PARTICIPANT_2` the object. This is the format
a station hands to the Vesida catalog and to anyone else who consumes optical tracking data.

```python
from opta_pipeline.tdm import write_tdm
write_tdm(tracklets, "night.tdm", originator="VESIDA", participant_1="NODE-01")
```

## Reference catalog (I-03)

Plate solving needs star positions. The default backend is a procedural catalog (deterministic,
offline, good for tests and first light). For real sky use `GaiaDR3Catalog.from_extract` with a local
Gaia DR3 extract; `GaiaDR3Catalog.write_extract` defines the on-disk format.
