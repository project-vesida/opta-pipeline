#!/usr/bin/env python3
"""Run the pipeline on a directory of I-01 FITS frames and write tracklets.

    python examples/run_on_fits.py frames/ --ra 135.0 --dec 45.0 --pixel-scale 23.93 \
        --out tracklets.json --tdm night.tdm

Needs only the runtime install (``pip install opta-pipeline``).  The plate solve is
blind: give the approximate field centre and plate scale, the matcher recovers the
rest against the reference catalog (procedural by default; see docs/interfaces.md
for Gaia DR3).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from opta_pipeline.config import PipelineConfig
from opta_pipeline.ingest import date_obs_to_mjd, load_fits_sequence
from opta_pipeline.match import PointingHint
from opta_pipeline.pipeline import FrameContext, run_pipeline
from opta_pipeline.tdm import write_tdm
from opta_pipeline.tracklet import tracklets_to_json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("frames_dir", type=Path, help="directory of I-01 FITS frames")
    ap.add_argument(
        "--ra", type=float, required=True, help="approx. field centre RA, deg J2000"
    )
    ap.add_argument(
        "--dec", type=float, required=True, help="approx. field centre Dec, deg J2000"
    )
    ap.add_argument(
        "--pixel-scale", type=float, default=None, help="arcsec/px (default: config)"
    )
    ap.add_argument(
        "--radius", type=float, default=2.0, help="catalog cone radius, deg"
    )
    ap.add_argument(
        "--mag-limit", type=float, default=12.0, help="faintest catalog star"
    )
    ap.add_argument(
        "--config", type=Path, default=None, help="pipeline YAML (default: shipped)"
    )
    ap.add_argument(
        "--no-stack", action="store_true", help="skip the blind track-and-stack search"
    )
    ap.add_argument(
        "--out", type=Path, default=Path("tracklets.json"), help="I-02 JSON output"
    )
    ap.add_argument(
        "--tdm", type=Path, default=None, help="also write a CCSDS TDM file"
    )
    ap.add_argument("--originator", default="VESIDA", help="TDM ORIGINATOR")
    args = ap.parse_args(argv)

    config = (
        PipelineConfig.from_yaml(args.config)
        if args.config
        else PipelineConfig.default()
    )
    if args.no_stack:
        from dataclasses import replace

        config = replace(config, stacking=replace(config.stacking, enabled=False))

    pixel_scale = args.pixel_scale
    if pixel_scale is None:
        import yaml

        from opta_pipeline.config import DEFAULTS_PATH

        with open(args.config or DEFAULTS_PATH) as fh:
            pixel_scale = float(yaml.safe_load(fh)["hardware"]["pixel_scale_arcsec"])

    hint = PointingHint(
        ra_deg=args.ra,
        dec_deg=args.dec,
        pixel_scale_arcsec=pixel_scale,
        radius_deg=args.radius,
        mag_limit=args.mag_limit,
    )

    frames = []
    node_id = "UNKNOWN"
    for i, (header, electrons, _path) in enumerate(load_fits_sequence(args.frames_dir)):
        node_id = header.node_id
        ctx = FrameContext(
            utc_mjd=date_obs_to_mjd(header.date_obs),
            frame_id=i,
            node_id=node_id,
            pointing=hint,
        )
        frames.append((electrons, ctx))
    if not frames:
        print(f"no FITS frames in {args.frames_dir}", file=sys.stderr)
        return 2

    tracklets = run_pipeline(frames, config)
    args.out.write_text(tracklets_to_json(tracklets))
    if args.tdm and tracklets:
        write_tdm(
            tracklets, args.tdm, originator=args.originator, participant_1=node_id
        )

    print(
        f"frames={len(frames)} tracklets={len(tracklets)} out={args.out}"
        + (f" tdm={args.tdm}" if args.tdm and tracklets else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
