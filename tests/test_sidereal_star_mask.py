"""Per-frame drift-tracking star masks (sidereal false-positive tracklets).

Regression suite for the sidereal star-mask gap (e2e findings §12; TODO.md,
open 2026-07-04): on a fixed mount the stars drift across the detector at the
sidereal rate, so

* temporal-median subtraction no longer models them — it leaves point-like
  *positive* residuals of ~30–50σ at each star's current position (measured
  on e2e scene D), and
* the frame-0 static star mask covers the wrong pixels for most of the pass.

Fast-mover trajectory hypotheses that merely *cross* several such residuals
accumulate ψ above the trials-corrected u* gate while dwelling on star pixels
for only ~10–28% of the frames — below any veto fraction that would still
pass real movers in star-dense fields, so no veto threshold can separate
them.  The fix is one layer down: per-frame star masks, built from each
frame's own astrometric star positions (WCS — sidereal drift is measured, not
fitted), enter the ψ/φ accumulation as invalid pixels (V = ∞ ⇒ φ = 0), so
the residual power never enters the statistic and the φ coverage keeps the
SNR map calibrated.

The drift rate used here is the physical sidereal rate for this geometry:
360.9856235°/86400 s ÷ plate scale — Earth's rotation and the hardware
profile, nothing fitted to the scene (design rule 10).
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace

import numpy as np
import pytest
from opta_model.hardware import VILTROX_85_F14_PRESET, compute_pixel_scale
from sensor_fixtures import SENSOR_SMALL

from opta_pipeline.astrometry import StarMatch, WCSSolution
from opta_pipeline.coarse_fine import blind_coarse_fine_search
from opta_pipeline.config import PipelineConfig
from opta_pipeline.likelihood import PsiPhiStacker
from opta_pipeline.pipeline import (
    FrameContext,
    VelocityPrior,
    _star_mask,
    _star_masks_per_frame,
    _trajectory_dominated_by_star,
    run_track_and_stack,
)
from opta_pipeline.synth import SatelliteSpec, StarSpec, generate_frame
from opta_pipeline.synth.catalog import catalog_stars_in_fov, star_field_at

_OPTICS = VILTROX_85_F14_PRESET
_PIXEL_SCALE = compute_pixel_scale(SENSOR_SMALL.pixel_size_um, _OPTICS.focal_length_mm)
_W = SENSOR_SMALL.resolution_h
_H = SENSOR_SMALL.resolution_v
_FPS = SENSOR_SMALL.frame_rate_hz

# Sidereal drift in image-plane px/s: Earth's rotation over the plate scale
# (fixed mount, equatorial pointing worst case).  ≈ 2.14 px/s here.
_SIDEREAL_PX_S = (360.9856235 / 86400.0) * 3600.0 / _PIXEL_SCALE

# 60 frames at 25 fps → T = 2.36 s → total drift ≈ 5.0 px: same regime as the
# failing e2e scene D (5.4 px over the pass, ≫ PSF FWHM 2 px), so the temporal
# median mis-models the stars the same way.
_N_FRAMES = 60
_MJD0 = 60000.0
_NODE = "NODE-SIDEREAL"

# Mover velocity for prior-path tests: LEO-plausible at this plate scale and
# far outside the sidereal band (e2e scene D: 26 px/s).
_V_OBJ_PX_S = -25.0


# ── unit: per-frame mask construction ───────────────────────────────────


def _matches_at(x: float, y: float) -> list[StarMatch]:
    return [StarMatch(x_px=x, y_px=y, ra_deg=10.0, dec_deg=20.0)]


# Base pixel positions for a small star field, spaced far enough apart that a
# nearest-neighbour link across a drifted pair cannot jump to the wrong star.
_FIELD_X = (20.0, 60.0, 100.0, 140.0, 180.0)
_FIELD_Y = (25.0, 15.0, 35.0, 20.0, 30.0)
# opta_validation.harness._SIDEREAL_RATE_DEG_S — the per-second RA advance the
# real-sky sidecar path applies (see the sidecar test below).
_SIDECAR_RA_RATE_DEG_S = 360.9856235 / 86400.0


def _star_field(shift_x: float, *, ra_offset_deg: float = 0.0) -> list[StarMatch]:
    """The whole field translated by ``shift_x`` px.

    ``ra_offset_deg`` re-labels every star's catalog RA, mimicking the
    real-sky sidecar path (``harness._drift_matches``) which rewrites RA per
    frame.  Pixel geometry is untouched by it.
    """
    return [
        StarMatch(
            x_px=x + shift_x,
            y_px=y,
            ra_deg=(10.0 + 0.5 * i + ra_offset_deg) % 360.0,
            dec_deg=20.0 + 0.1 * i,
        )
        for i, (x, y) in enumerate(zip(_FIELD_X, _FIELD_Y))
    ]


class TestStarMasksPerFrame:
    def test_masks_track_the_drift(self) -> None:
        """Frame k's mask covers frame k's star position, not frame 0's."""
        shape = (50, 80)
        drift = [_matches_at(20.0 + 2.0 * k, 25.0) for k in range(6)]
        masks = _star_masks_per_frame(shape, drift, radius_px=3)
        assert masks is not None and len(masks) == 6
        assert masks[5][25, 30]  # star at x=30 in frame 5
        assert not masks[5][25, 20]  # frame-0 position no longer masked
        assert masks[0][25, 20]

    def test_unsolved_frame_borrows_nearest_solved(self) -> None:
        shape = (50, 80)
        per_frame = [
            _matches_at(20.0, 25.0),
            [],  # plate solve failed
            _matches_at(30.0, 25.0),
        ]
        masks = _star_masks_per_frame(shape, per_frame, radius_px=3)
        assert masks is not None
        # frame 1 borrows a neighbour (either side is one frame away).
        assert masks[1][25, 20] or masks[1][25, 30]

    def test_borrow_is_capped_at_the_measured_drift_distance(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A gap of failed solves longer than radius/drift is left unmasked.

        Borrowing the nearest solved frame's positions is only justified while
        the drift stays inside the mask radius.  Here the field moves 1 px per
        frame against a 5 px radius, so the cap is 5 frames: frames within 5
        of a solve still borrow, the ones further out are left unmasked, and
        the degradation is logged rather than silently stamping discs on blank
        sky.
        """
        shape = (60, 260)
        radius = 5
        drift_px_per_frame = 1.0
        n = 15
        solved_at = (0, 1, n - 1)
        per_frame = [
            _star_field(drift_px_per_frame * k) if k in solved_at else []
            for k in range(n)
        ]

        with caplog.at_level(logging.WARNING, logger="opta_pipeline.pipeline"):
            masks = _star_masks_per_frame(shape, per_frame, radius_px=radius)
        assert masks is not None and len(masks) == n

        cap = int(radius / drift_px_per_frame)  # 5 frames
        for k in range(n):
            nearest = min(abs(k - s) for s in solved_at)
            if nearest <= cap:
                assert masks[k].any(), f"frame {k} ({nearest} away) lost its mask"
            else:
                assert not masks[k].any(), f"frame {k} ({nearest} away) borrowed"
        assert "left UNMASKED" in caplog.text

    def test_drift_is_measured_from_pixels_not_catalog_coordinates(self) -> None:
        """The real-sky sidecar path re-labels catalog RA on every frame.

        ``opta_validation.harness._drift_matches`` hands each frame its own
        ``StarMatch`` list with ``ra_deg = (ra + sidereal_rate·dt) % 360`` —
        the same physical stars, never the same RA twice.  A drift estimator
        that joined stars on catalog coordinates would find zero shared stars
        across 15 solved frames, report "unmeasured", and collapse to the
        1-frame cap on exactly the partially-solved real night the cap exists
        to protect.  Identical pixel geometry to the test above, so the cap
        must come out identical.
        """
        shape = (60, 260)
        radius = 5
        n = 15
        solved_at = (0, 1, n - 1)
        dt = 1.0 / 25.0  # 25 fps
        per_frame = [
            _star_field(
                1.0 * k, ra_offset_deg=_SIDECAR_RA_RATE_DEG_S * (k * dt)
            )
            if k in solved_at
            else []
            for k in range(n)
        ]
        # Precondition: no catalog coordinate is shared by any two frames.
        keys = [
            {(m.ra_deg, m.dec_deg) for m in per_frame[s]} for s in solved_at
        ]
        assert not keys[0] & keys[1] and not keys[1] & keys[2]

        masks = _star_masks_per_frame(shape, per_frame, radius_px=radius)
        assert masks is not None
        for k in range(n):
            nearest = min(abs(k - s) for s in solved_at)
            assert masks[k].any() == (nearest <= 5), f"frame {k} at {nearest}"

    def test_borrow_cap_scales_with_the_measured_drift(self) -> None:
        """A stationary mount (zero measured drift) borrows without limit."""
        shape = (60, 260)
        n = 15
        static = [_star_field(0.0) if k in (0, n - 1) else [] for k in range(n)]
        masks = _star_masks_per_frame(shape, static, radius_px=5)
        assert masks is not None
        assert all(m.any() for m in masks)

    def test_unmeasurable_drift_falls_back_to_adjacent_only(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One solved frame ⇒ no drift estimate ⇒ conservative 1-frame cap."""
        shape = (60, 260)
        per_frame: list[list[StarMatch]] = [[] for _ in range(6)]
        per_frame[0] = _star_field(0.0)
        with caplog.at_level(logging.WARNING, logger="opta_pipeline.pipeline"):
            masks = _star_masks_per_frame(shape, per_frame, radius_px=5)
        assert masks is not None
        assert masks[0].any() and masks[1].any()
        assert not any(m.any() for m in masks[2:])
        assert "left UNMASKED" in caplog.text

    def test_too_few_linked_stars_is_unmeasured(self) -> None:
        """Below _MIN_DRIFT_PAIRS links a pair is a coin flip, not a sample."""
        shape = (60, 260)
        n = 8
        # Two stars per solved frame: enough to pair, too few to trust.
        pair = [
            [
                StarMatch(x_px=20.0 + k, y_px=25.0, ra_deg=10.0, dec_deg=20.0),
                StarMatch(x_px=90.0 + k, y_px=25.0, ra_deg=11.0, dec_deg=20.0),
            ]
            if k in (0, 1, n - 1)
            else []
            for k in range(n)
        ]
        masks = _star_masks_per_frame(shape, pair, radius_px=5)
        assert masks is not None
        # Unmeasured ⇒ 1-frame cap: only frames adjacent to a solve keep a mask.
        for k in range(n):
            nearest = min(abs(k - s) for s in (0, 1, n - 1))
            assert masks[k].any() == (nearest <= 1), f"frame {k} at {nearest}"

    def test_disabled_or_empty_returns_none(self) -> None:
        assert _star_masks_per_frame((50, 80), [_matches_at(1, 1)], 0) is None
        assert _star_masks_per_frame((50, 80), [[], []], 5) is None
        assert _star_masks_per_frame((50, 80), [], 5) is None

    def test_no_solved_frame_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Masking requested but NO frame has star matches → warn, not silent.

        Returning None here means the ψ/φ stack runs with no star masks at
        all — the sidereal FP-tracklet regime (e2e findings §12) — so the
        degradation must be visible in the logs.
        """
        with caplog.at_level(logging.WARNING, logger="opta_pipeline.pipeline"):
            result = _star_masks_per_frame((50, 80), [[], []], 5)
        assert result is None
        assert "no frame has star matches" in caplog.text

    def test_disabled_masking_does_not_warn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """radius_px <= 0 is an intentional off-switch — no warning."""
        with caplog.at_level(logging.WARNING, logger="opta_pipeline.pipeline"):
            assert _star_masks_per_frame((50, 80), [[], []], 0) is None
        assert caplog.text == ""

    def test_stamped_mask_matches_reference_disc(self) -> None:
        """The stamped implementation equals the brute-force disc test."""
        shape = (40, 60)
        matches = [
            StarMatch(x_px=10.3, y_px=8.7, ra_deg=0.0, dec_deg=0.0),
            StarMatch(x_px=-2.0, y_px=20.0, ra_deg=0.0, dec_deg=0.0),  # clipped
            StarMatch(x_px=59.5, y_px=39.5, ra_deg=0.0, dec_deg=0.0),  # corner
        ]
        r = 4
        got = _star_mask(shape, matches, r)
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        want = np.zeros(shape, dtype=bool)
        for m in matches:
            want |= (xx - m.x_px) ** 2 + (yy - m.y_px) ** 2 <= float(r) ** 2
        assert np.array_equal(got, want)


class TestVetoWithPerFrameMasks:
    _TIMES = [(k - 19.5) / 25.0 for k in range(40)]

    @staticmethod
    def _disc(shape, cx, cy, r):
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r

    def test_drift_locked_hypothesis_is_vetoed(self) -> None:
        """A hypothesis riding a *drifting* star dwells on per-frame masks.

        With the static frame-0 mask the star walks off its own disc once
        the drift exceeds the radius; per-frame masks keep covering it.
        """
        shape = (300, 400)
        drift = 12.0  # px/s: leaves an r=3 static disc within 1/3 of the pass
        masks = [
            self._disc(shape, 200.0 + drift * t, 150.0, 3) for t in self._TIMES
        ]
        assert _trajectory_dominated_by_star(
            200.0, 150.0, drift, 0.0, self._TIMES, masks, max_frac=0.5
        )
        static = self._disc(shape, 200.0, 150.0, 3)
        assert not _trajectory_dominated_by_star(
            200.0, 150.0, drift, 0.0, self._TIMES, static, max_frac=0.5
        )

    def test_transiting_mover_survives(self) -> None:
        shape = (300, 400)
        masks = [
            self._disc(shape, 200.0 + 2.0 * t, 150.0, 3) for t in self._TIMES
        ]
        assert not _trajectory_dominated_by_star(
            200.0, 150.0, 200.0, 0.0, self._TIMES, masks, max_frac=0.5
        )

    def test_misaligned_mask_list_raises(self) -> None:
        shape = (300, 400)
        masks = [self._disc(shape, 200.0, 150.0, 3)] * 3  # != 40 epochs
        with pytest.raises(ValueError, match="align"):
            _trajectory_dominated_by_star(
                200.0, 150.0, 0.0, 0.0, self._TIMES, masks, max_frac=0.5
            )


# ── mechanism: masked ψ/φ removes the drifting-star residual ────────────


def _drifting_star_frames(
    rng: np.random.Generator,
    *,
    shape: tuple[int, int] = (120, 160),
    n_frames: int = _N_FRAMES,
    flux_e: float = 20_000.0,
    sigma_px: float = 0.85,
    noise_rms: float = 2.0,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Calibrated-equivalent frames: one bright star drifting at sidereal rate.

    Pixel-space Gaussian blobs (no sky projection involved), zero-mean
    Gaussian noise — the minimal scene in which temporal-median subtraction
    leaves the drift residual.  Returns (frames, per-frame masks, times).
    """
    h, w = shape
    times = [(k - (n_frames - 1) / 2.0) / _FPS for k in range(n_frames)]
    x0, y0 = w / 2.0, h / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    frames, masks = [], []
    for t in times:
        x = x0 + _SIDEREAL_PX_S * t
        star = flux_e / (2 * math.pi * sigma_px**2) * np.exp(
            -((xx - x) ** 2 + (yy - y0) ** 2) / (2 * sigma_px**2)
        )
        frames.append(star + rng.normal(0.0, noise_rms, shape))
        masks.append((xx - x) ** 2 + (yy - y0) ** 2 <= 5.0**2)
    return frames, masks, times


class TestMaskedPsiPhiMechanism:
    def test_masks_remove_drift_residual_from_snr_map(self) -> None:
        """Without masks the median residual stacks far above u*; with
        per-frame masks the map is pure noise below u*."""
        rng = np.random.default_rng(42)
        frames, masks, times = _drifting_star_frames(rng)
        stacker = PsiPhiStacker.from_prior(
            0.0, _V_OBJ_PX_S, half_width_px_s=1.5, step_px_s=0.5
        )
        n_hyp = stacker.vx_grid.size * stacker.vy_grid.size
        u_star = math.sqrt(2.0 * math.log(120 * 160 * n_hyp)) + 0.5

        unmasked = stacker.stack(
            frames, 2.0, times, subtract_temporal_median=True
        )
        masked = stacker.stack(
            frames, 2.0, times, masks=masks, subtract_temporal_median=True
        )
        # The residual is the dominant feature of the unmasked map…
        assert unmasked.snr_map.max() > u_star
        # …and entirely absent from the masked one.
        assert masked.snr_map.max() < u_star
        assert masked.snr_map.max() < 0.5 * unmasked.snr_map.max()

    def test_blind_search_seeds_nothing_on_masked_residual(self) -> None:
        """The coarse-to-fine blind path honours per-frame masks end to end."""
        rng = np.random.default_rng(7)
        frames, masks, times = _drifting_star_frames(rng)
        kwargs = dict(
            v_max_px_s=10.0,
            coarse_step_px_s=5.0,
            prethreshold_sigma=4.0,
            far_margin_sigma=0.5,
        )
        without = blind_coarse_fine_search(frames, 2.0, times, **kwargs)
        with_masks = blind_coarse_fine_search(
            frames, 2.0, times, masks=masks, **kwargs
        )
        assert without.candidates  # the scene is adversarial: residual seeds
        assert with_masks.candidates == ()


# ── pipeline: sidereal scene through run_track_and_stack ────────────────


def _wcs() -> WCSSolution:
    scale_deg = _PIXEL_SCALE / 3600.0
    return WCSSolution(
        crpix1=_W / 2.0,
        crpix2=_H / 2.0,
        crval1=135.0,
        crval2=45.0,
        cd1_1=scale_deg,
        cd1_2=0.0,
        cd2_1=0.0,
        cd2_2=scale_deg,
        rms_arcsec=0.0,
        n_stars=0,
    )


def _config() -> PipelineConfig:
    cfg = PipelineConfig.default()
    return replace(
        cfg,
        detection=replace(cfg.detection, min_streak_pixels=3),
        tracklet=replace(cfg.tracklet, linear_fit_residual_arcsec=20.0),
        stacking=replace(cfg.stacking, enabled=True),
    )


def _prior() -> VelocityPrior:
    return VelocityPrior(
        vx_px_s=0.0,
        vy_px_s=_V_OBJ_PX_S,
        half_width_px_s=1.5,
        step_px_s=0.5,
    )


def _sidereal_frame_pairs(with_satellite: bool, seed0: int) -> list:
    """Star field drifting at the sidereal rate (+ optional mover).

    Mirrors the e2e --sidereal-drift harness: the stars (and their sidecar
    ``star_matches``) move together at the known sidereal rate while their
    RA/Dec stay fixed, exactly what a fixed mount sees; each frame's
    supplied matches are that frame's own astrometry (production: per-frame
    plate solve).
    """
    wcs = _wcs()
    base_matches = star_field_at(wcs, _W, _H, mag_limit=14.0)
    catalog_stars = catalog_stars_in_fov(wcs, _W, _H, mag_limit=11.0)
    if len(catalog_stars) < 6:
        catalog_stars = catalog_stars_in_fov(wcs, _W, _H, mag_limit=14.0)[:12]
    star_specs = StarSpec.from_catalog(wcs, catalog_stars)

    reference_s = ((_N_FRAMES - 1) / 2.0) / _FPS
    angular_velocity_deg_s = abs(_V_OBJ_PX_S) * _PIXEL_SCALE / 3600.0

    pairs = []
    for frame_id in range(_N_FRAMES):
        t_s = frame_id / _FPS
        dx = _SIDEREAL_PX_S * (t_s - reference_s)
        specs = [replace(s, x=s.x + dx) for s in star_specs]
        matches = [replace(m, x_px=m.x_px + dx) for m in base_matches]
        sats = []
        if with_satellite:
            sats.append(
                SatelliteSpec(
                    magnitude=12.0,
                    angular_velocity_deg_s=angular_velocity_deg_s,
                    x_center=_W / 2.0,
                    y_center=_H / 2.0 + _V_OBJ_PX_S * (t_s - reference_s),
                    angle_deg=-90.0,
                )
            )
        synth = generate_frame(
            SENSOR_SMALL,
            _OPTICS,
            sky_mag_arcsec2=21.0,
            elevation_deg=45.0,
            satellites=sats,
            stars=specs,
            rng=np.random.default_rng(seed0 + frame_id),
        )
        ctx = FrameContext(
            star_matches=matches,
            utc_mjd=_MJD0 + t_s / 86400.0,
            frame_id=frame_id,
            node_id=_NODE,
        )
        pairs.append((synth.data_float, ctx))
    return pairs


class TestSiderealPipeline:
    """run_track_and_stack on a drifting star field (prior path, as e2e)."""

    def test_drifting_stars_alone_emit_no_tracklets(self) -> None:
        """THE regression for the §12 sidereal FP tracklets: star drift +
        median residuals + a fast-mover velocity prior must not mint
        tracklets (e2e scene D emitted 12 of these from exactly this
        geometry before per-frame masks)."""
        result = run_track_and_stack(
            _sidereal_frame_pairs(False, 3100),
            config=_config(),
            velocity_prior=_prior(),
        )
        assert result.stacked_detections == ()
        assert result.tracklets == ()

    def test_mover_recovered_among_drifting_stars(self) -> None:
        """Completeness guard: the per-frame masks must not eat a real mover
        crossing the drifting star field (it loses only the few frames it
        spends on masked pixels)."""
        result = run_track_and_stack(
            _sidereal_frame_pairs(True, 3300),
            config=_config(),
            velocity_prior=_prior(),
        )
        assert result.stack_result.best_vy_px_s == pytest.approx(
            _V_OBJ_PX_S, abs=1.0
        )
        assert len(result.tracklets) >= 1
        best = max(result.tracklets, key=lambda t: len(t.points))
        expected_rate = abs(_V_OBJ_PX_S) * _PIXEL_SCALE
        assert math.hypot(
            best.ra_rate_arcsec_s, best.dec_rate_arcsec_s
        ) == pytest.approx(expected_rate, rel=0.10)
