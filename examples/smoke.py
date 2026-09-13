#!/usr/bin/env python3
"""Smoke test on synthetic frames: synth -> detect -> astrometry -> tracklet -> TDM.

Needs the synth extra:  pip install 'opta-pipeline[synth]'
Prints ``detected=True`` when at least one tracklet forms.
"""

from __future__ import annotations

import math
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

try:
    from opta_model.hardware_catalog import build_node_from_profile
except ImportError:
    sys.exit("this example needs opta-model: pip install 'opta-pipeline[synth]'")

from opta_pipeline.astrometry import StarMatch
from opta_pipeline.pipeline import FrameContext, run_frame
from opta_pipeline.synth import SatelliteSpec, StarSpec, generate_frame
from opta_pipeline.tdm import write_tdm
from opta_pipeline.tracklet import link_detections

# The test-suite node: IMX585 pixel parameters behind an 85 mm lens, 400x300 px
# window, so the run finishes in seconds.  Any catalog profile works here.
node = build_node_from_profile("pipeline_toy")
sensor = replace(node.sensor, resolution_h=400, resolution_v=300)
optics = node.optics
pixel_scale = node.pixel_scale_arcsec  # arcsec/px

W, H = sensor.resolution_h, sensor.resolution_v
CX, CY = W / 2.0, H / 2.0
RA0, DEC0 = 135.0, 45.0
N_FRAMES = 10
DT_MJD = (1.0 / sensor.frame_rate_hz) / 86400.0
STAR_XS, STAR_YS = [60.0, 140.0, 260.0, 340.0], [60.0, 240.0]


def px_to_sky(x: float, y: float) -> tuple[float, float]:
    """Exact TAN deprojection for the ground-truth WCS."""
    xi = math.radians(pixel_scale * (x - CX) / 3600.0)
    eta = math.radians(pixel_scale * (y - CY) / 3600.0)
    d0 = math.radians(DEC0)
    denom = math.cos(d0) - eta * math.sin(d0)
    ra = RA0 + math.degrees(math.atan2(xi, denom))
    dec = math.degrees(
        math.atan2(math.sin(d0) + eta * math.cos(d0), math.hypot(xi, denom))
    )
    return ra, dec


matches = [
    StarMatch(x_px=x, y_px=y, ra_deg=px_to_sky(x, y)[0], dec_deg=px_to_sky(x, y)[1])
    for x in STAR_XS
    for y in STAR_YS
]

frame_dets = []
peak_snr = 0.0
for i in range(N_FRAMES):
    cx = CX - (N_FRAMES / 2.0) * 8.0 + i * 8.0
    synth = generate_frame(
        sensor=sensor,
        optics=optics,
        sky_mag_arcsec2=21.0,
        elevation_deg=45.0,
        satellites=[
            SatelliteSpec(
                magnitude=9.0, angular_velocity_deg_s=0.5, x_center=cx, y_center=CY
            )
        ],
        stars=[StarSpec(magnitude=9.5, x=x, y=y) for x in STAR_XS for y in STAR_YS],
        rng=np.random.default_rng(2000 + i),
    )
    ctx = FrameContext(
        star_matches=matches,
        utc_mjd=60000.0 + i * DT_MJD,
        frame_id=i,
        node_id="SYNTH-01",
    )
    fd = run_frame(synth.data, ctx)
    if fd is None:
        continue
    frame_dets.append(fd)
    peak_snr = max(peak_snr, max((d.detection.snr for d in fd.detections), default=0.0))

tracklets = link_detections(
    frame_dets,
    max_sep_arcsec=300.0,
    min_points=3,
    max_rms_arcsec=50.0,
    max_gap_frames=2,
)
out: Path | None = None
if tracklets:
    out = Path(tempfile.mkdtemp()) / "smoke.tdm"
    write_tdm(tracklets, out, originator="VESIDA", participant_1="SYNTH-01")

print(
    f"detected={bool(tracklets)} frames_with_detections={len(frame_dets)}/{N_FRAMES} "
    f"n_tracklets={len(tracklets)} peak_snr={peak_snr:.1f} tdm={out}"
)
