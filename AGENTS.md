# AGENTS.md

Read this before working in `opta-pipeline`. Humans: `README.md` covers install and use.

## Ground rules

1. Code and tests are the source of truth; if a docstring or this file disagrees with the code, fix the doc in the same commit.
2. The runtime path (`ingest` → `calibrate` → `detect` → `astrometry` → `tracklet`/`likelihood` → `tdm`) must import nothing from `opta_model`. Only `synth/` and profile-name lookups in `config.py` may, behind the `[synth]` extra. CI's `core` job enforces this.
3. Never quote a pinned number (SNR, gate value, completeness) without the command that reproduces it. Say which SNR: pixel `peak_snr` and matched-filter component SNR differ by 2–3x on faint trails.
4. Verify before reporting: run the commands below for anything you touched.

## Modules

| Module | Role |
|---|---|
| `pipeline.py` | `run_frame`, `run_pipeline`, `run_track_and_stack`, `FrameContext`, `PointingHint` |
| `ingest.py` | I-01 FITS loading and header validation; directory and astrometry.net `corr.fits` helpers |
| `ingest_video.py` | Video/PNG frame sequences to I-01 FITS (needs ffmpeg on PATH; `[video]`) |
| `calibrate.py` | Dark, flat, mesh background |
| `detect.py` | Source extraction, centroiding, streak classification |
| `astrometry.py` | WCS fit (TAN + SIP), blind solve entry points |
| `catalog.py`, `procedural_catalog.py`, `match.py` | Reference-catalog backends (procedural default, Gaia DR3 extract) and star matching |
| `stack.py`, `likelihood.py`, `coarse_fine.py` | Shift-and-add reference, ψ/φ matched-filter track-and-stack, blind coarse-to-fine velocity search |
| `tracklet.py` | Hungarian linking, linear-motion QC, I-02 records |
| `tdm.py` | CCSDS TDM 503.0-B-2 export |
| `config.py` | `PipelineConfig` from `configs/pipeline_defaults.yaml`; gates derive from `hardware.frame_rate_hz` and `hardware.pixel_scale_arcsec` |
| `synth/` | Synthetic frames and SGP4 passes from opta-model physics (`[synth]`) |

## Commands

```bash
pip install -e '.[dev]' -c constraints.txt
ruff check src tests examples && pyright
python -m pytest tests -q -m "not slow" -n 4 --dist loadfile   # full fast suite
python examples/smoke.py                                       # detected=True
```

Without opta-model installed, `pytest tests` runs only the core modules; `conftest.py` reports how many synth-dependent modules it skipped.

## Gotchas

- Test sensors are 400x300 px variants of the IMX585 (`tests/sensor_fixtures.py`); full-size frames make the suite 20x slower.
- `run_pipeline` always runs both the per-frame chain and the blind stacked search when `stacking.enabled`; tests of the per-frame path use `per_frame_only_config()`.
- xdist must use `--dist loadfile`, otherwise module-scoped blind-search fixtures are rebuilt in every worker.
