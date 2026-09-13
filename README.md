# opta-pipeline

Part of [Project Vesida](https://github.com/project-vesida). Apache-2.0.

Detection pipeline of the Optical Transit Array (OpTA): camera frames in, astrometric satellite
tracklets out. It runs on recorded frames today and is built to run on the array node itself.

```
frames (I-01 FITS) → calibrate → detect / track-and-stack → astrometry → tracklets (I-02 JSON, CCSDS TDM)
```

- **Calibrate**: dark, flat, mesh background.
- **Detect**: source extraction, centroiding, streak classification.
- **Track-and-stack**: ψ/φ matched-filter search over a velocity grid for movers too faint to see in a single frame, with a coarse-to-fine blind search.
- **Astrometry**: TAN + SIP plate solve against a reference catalog, rolling-shutter correction.
- **Tracklets**: Hungarian frame-to-frame linking with linear-motion quality control.
- **Export**: I-02 JSON records and CCSDS Tracking Data Messages.

The runtime depends only on numpy, scipy, astropy, pandas, matplotlib and PyYAML. The physics model
behind the synthetic-frame renderer lives in [opta-model](https://github.com/project-vesida/opta-model)
and is an optional extra.

## Install

Python 3.11 or newer.

```bash
pip install opta-pipeline                 # runtime
pip install 'opta-pipeline[synth]'        # + synthetic frames (pulls opta-model)
pip install 'opta-pipeline[video]'        # + video/PNG ingest (needs ffmpeg on PATH)
```

From a checkout, for development:

```bash
pip install -e '.[dev]' -c constraints.txt
```

## Use

Run on a directory of I-01 FITS frames and write tracklets:

```bash
python examples/run_on_fits.py frames/ --ra 135.0 --dec 45.0 --out tracklets.json --tdm night.tdm
```

Or from Python:

```python
from opta_pipeline import PipelineConfig, run_pipeline
from opta_pipeline.ingest import date_obs_to_mjd, load_fits_sequence
from opta_pipeline.match import PointingHint
from opta_pipeline.pipeline import FrameContext
from opta_pipeline.tdm import write_tdm

hint = PointingHint(ra_deg=135.0, dec_deg=45.0, pixel_scale_arcsec=23.93)
frames = []
for i, (header, electrons, path) in enumerate(load_fits_sequence("frames/")):
    ctx = FrameContext(utc_mjd=date_obs_to_mjd(header.date_obs), frame_id=i,
                       node_id=header.node_id, pointing=hint)
    frames.append((electrons, ctx))

tracklets = run_pipeline(frames, PipelineConfig.default())
write_tdm(tracklets, "night.tdm", originator="VESIDA", participant_1=header.node_id)
```

Smoke test on synthetic frames (needs `[synth]`):

```bash
python examples/smoke.py      # prints detected=True ... n_tracklets=1
```

Configuration is one YAML file, `src/opta_pipeline/configs/pipeline_defaults.yaml`. The two
scale-dependent gates derive from `hardware.frame_rate_hz` and `hardware.pixel_scale_arcsec`, so a
different camera needs those two numbers changed and nothing else.

## Interfaces

[docs/interfaces.md](docs/interfaces.md) documents the frame header contract (I-01), the tracklet
record (I-02), and the CCSDS TDM export. These are the seams a node, a different pipeline, or the
Vesida catalog plug into.

## Roadmap

The next integration target is the OpTA reference hardware defined in
[opta-hardware](https://github.com/project-vesida/opta-hardware).

Toward on-node operation: streaming ingest instead of whole-pass batches, float32 and FFT kernels,
a backend seam for accelerators, and the array agent that schedules captures and uploads TDM
([vesida-agent](https://github.com/project-vesida/vesida-agent)). Issues on this repository track
the work.

## Develop

```bash
ruff check src tests examples && pyright
python -m pytest tests -q -m "not slow" -n 4 --dist loadfile
```

Most tests render synthetic frames and need `[dev]`. Without opta-model installed the suite runs the
core modules only and says so. `AGENTS.md` carries the module map and gotchas for coding agents.
