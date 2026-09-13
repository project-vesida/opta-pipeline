"""Scene-domain pointing jitter: the sky moves, the detector does not.

``PointingJitter`` exists to replace a post-render whole-frame resample
(``opta_validation.corruption.pointing_jitter``) that had two defects: it
translated the instrumental signature along with the scene, and bilinear
interpolation of an already-noisy frame spatially correlated and attenuated the
noise.  These tests pin the three properties that make the scene-domain stage
the correct tool:

* the static detector signature stays at fixed detector pixels while a source
  centroid moves by exactly the drawn offset;
* the noise field is untouched (its variance matches the no-jitter case),
  unlike a post-render resample — pinned here side by side;
* ``pointing_jitter=None`` is an exact no-op, so every pre-existing caller
  keeps byte-identical frames.

Uses SENSOR_SMALL (400×300); physics still derives from opta_model.radiometry.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.ndimage import shift as ndimage_shift
from sensor_fixtures import OPTICS_DEFAULT, SENSOR_SMALL

from opta_pipeline.synth import (
    PointingJitter,
    SatelliteSpec,
    StarSpec,
    build_detector_model,
    generate_frame,
)
from opta_pipeline.synth.render import (
    default_pipeline,
    render_frame,
    stage_pointing_jitter,
)

SENSOR = SENSOR_SMALL
OPTICS = OPTICS_DEFAULT

#: A hot pixel far from every injected source, so its excess is unambiguous.
HOT_Y, HOT_X = 40, 340
#: Dark-current rate for that pixel (e⁻/s): ~4000 e⁻ in one 0.04 s frame.
HOT_RATE_E_S = 1.0e5


def _clean_detector():
    """Detector with a flat response and no noise structure — a blank slate."""
    det = build_detector_model(
        SENSOR,
        seed=0,
        prnu_pct=0.0,
        vignetting_corner_factor=1.0,
        bias_offset_e=0.0,
        bias_fpn_rms_e=0.0,
        hot_pixel_fraction=0.0,
        dead_pixel_fraction=0.0,
    )
    return dataclasses.replace(
        det, dark_current_e_s=np.zeros(det.shape, dtype=np.float32)
    )


def _defective_detector():
    """Blank slate plus one very hot pixel and one dead column in the flat."""
    det = _clean_detector()
    dark = np.zeros(det.shape, dtype=np.float32)
    dark[HOT_Y, HOT_X] = HOT_RATE_E_S
    flat = np.ones(det.shape, dtype=np.float32)
    flat[:, 12] = 1e-3  # dead column: a strong *multiplicative* signature
    return dataclasses.replace(det, dark_current_e_s=dark, flat=flat)


def _bright_star(x: float, y: float) -> StarSpec:
    """Free-injection star bright enough to centroid to ~0.01 px."""
    return StarSpec(magnitude=8.0, x=x, y=y, signal_e=2.0e6)


def _centroid(image: np.ndarray, x0: float, y0: float, half: int = 8):
    """Background-subtracted flux-weighted centroid in a window about (x0, y0)."""
    xi, yi = int(round(x0)), int(round(y0))
    win = image[yi - half : yi + half + 1, xi - half : xi + half + 1]
    win = np.clip(win - np.median(image), 0.0, None)
    yy, xx = np.mgrid[yi - half : yi + half + 1, xi - half : xi + half + 1]
    tot = win.sum()
    return float((xx * win).sum() / tot), float((yy * win).sum() / tot)


# ---------------------------------------------------------------------------
# The offset model
# ---------------------------------------------------------------------------


class TestPointingJitterModel:
    def test_zero_rms_is_exact_noop(self) -> None:
        pj = PointingJitter(rms_px=0.0, seed=5)
        assert all(pj.offset_px(k) == (0.0, 0.0) for k in range(10))

    def test_negative_rms_rejected(self) -> None:
        with pytest.raises(ValueError, match="rms_px"):
            PointingJitter(rms_px=-0.1)

    def test_offset_depends_only_on_seed_and_index(self) -> None:
        """Order-independence: frame 7 alone == frame 7 of a rendered sequence."""
        pj = PointingJitter(rms_px=0.7, seed=3)
        standalone = pj.offset_px(7)
        in_sequence = [
            PointingJitter(rms_px=0.7, seed=3).offset_px(k) for k in range(8)
        ]
        assert standalone == in_sequence[7]

    def test_offsets_vary_per_frame(self) -> None:
        pj = PointingJitter(rms_px=1.0, seed=3)
        offsets = {pj.offset_px(k) for k in range(20)}
        assert len(offsets) == 20

    def test_offset_statistics_match_rms(self) -> None:
        pj = PointingJitter(rms_px=0.8, seed=11)
        draws = np.array([pj.offset_px(k) for k in range(4000)])
        assert draws.mean() == pytest.approx(0.0, abs=0.05)
        assert draws.std() == pytest.approx(0.8, rel=0.05)

    def test_different_seeds_differ(self) -> None:
        a = [PointingJitter(rms_px=0.5, seed=1).offset_px(k) for k in range(5)]
        b = [PointingJitter(rms_px=0.5, seed=2).offset_px(k) for k in range(5)]
        assert a != b


# ---------------------------------------------------------------------------
# (a) detector signature fixed, scene moves
# ---------------------------------------------------------------------------


class TestDetectorSignatureStaysFixed:
    """The defect the TODO item names: the signature must NOT move with the sky."""

    def _frames(self, pj: PointingJitter | None, n: int = 6):
        det = _defective_detector()
        return [
            render_frame(
                SENSOR,
                OPTICS,
                det,
                sky_mag_arcsec2=21.0,
                stars=[_bright_star(200.0, 150.0)],
                psf_fwhm_px=3.0,
                pointing_jitter=pj,
                frame_index=k,
                rng=np.random.default_rng(1000 + k),
            )
            for k in range(n)
        ]

    def test_hot_pixel_and_dead_column_do_not_move(self) -> None:
        frames = self._frames(PointingJitter(rms_px=1.5, seed=21))
        for f in frames:
            img = f.data_float
            # The hot pixel sits at its detector address in every frame...
            assert img[HOT_Y, HOT_X] > 0.5 * HOT_RATE_E_S / SENSOR.frame_rate_hz
            # ...and its neighbours never light up (i.e. it never translated).
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                assert img[HOT_Y + dy, HOT_X + dx] < 100.0
            # The dead column stays exactly at column 12.
            assert img[:, 12].mean() < img[:, 11].mean()
            assert img[:, 12].mean() < img[:, 13].mean()

    def test_source_centroid_moves_by_the_drawn_offset(self) -> None:
        pj = PointingJitter(rms_px=1.5, seed=21)
        frames = self._frames(pj)
        for k, f in enumerate(frames):
            dx, dy = pj.offset_px(k)
            cx, cy = _centroid(f.data_float, 200.0 + dx, 150.0 + dy)
            assert cx == pytest.approx(200.0 + dx, abs=0.05)
            assert cy == pytest.approx(150.0 + dy, abs=0.05)
        # Sanity: the offsets under test are real motion, not float noise.
        assert max(abs(pj.offset_px(k)[0]) for k in range(6)) > 0.5

    def test_satellite_streak_translates_too(self) -> None:
        """Whole-field rule: the mover shares the scene offset with the stars."""
        pj = PointingJitter(rms_px=2.0, seed=4)
        det = _clean_detector()
        sat = SatelliteSpec(
            magnitude=8.0, angular_velocity_deg_s=0.2,
            x_center=200.0, y_center=150.0, angle_deg=0.0, signal_e=2.0e6,
        )
        for k in (0, 1, 2):
            dx, dy = pj.offset_px(k)
            f = render_frame(
                SENSOR, OPTICS, det, satellites=[sat], psf_fwhm_px=3.0,
                pointing_jitter=pj, frame_index=k,
                rng=np.random.default_rng(5),
            )
            truth = f.satellites[0]
            assert truth.x_center == pytest.approx(200.0 + dx, abs=1e-9)
            assert truth.y_center == pytest.approx(150.0 + dy, abs=1e-9)
            cx, cy = _centroid(f.data_float, 200.0 + dx, 150.0 + dy, half=10)
            assert cy == pytest.approx(150.0 + dy, abs=0.05)


# ---------------------------------------------------------------------------
# (b) reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def _sequence(self, seed: int, rng_seed: int = 7):
        pj = PointingJitter(rms_px=1.0, seed=seed)
        det = _defective_detector()
        return [
            render_frame(
                SENSOR, OPTICS, det, stars=[_bright_star(200.0, 150.0)],
                pointing_jitter=pj, frame_index=k,
                rng=np.random.default_rng(rng_seed + k),
            ).data_float
            for k in range(4)
        ]

    def test_same_seed_gives_identical_sequence(self) -> None:
        a = self._sequence(seed=17)
        b = self._sequence(seed=17)
        assert all(np.array_equal(x, y) for x, y in zip(a, b))

    def test_different_seed_gives_different_sequence(self) -> None:
        a = self._sequence(seed=17)
        b = self._sequence(seed=18)
        assert not any(np.array_equal(x, y) for x, y in zip(a, b))

    def test_clean_path_reproducible(self) -> None:
        def seq():
            pj = PointingJitter(rms_px=1.0, seed=31)
            return [
                generate_frame(
                    SENSOR, OPTICS, stars=[_bright_star(200.0, 150.0)],
                    rng=np.random.default_rng(3 + k),
                    pointing_jitter=pj, frame_index=k,
                ).data_float
                for k in range(3)
            ]

        assert all(np.array_equal(x, y) for x, y in zip(seq(), seq()))


# ---------------------------------------------------------------------------
# (c) no-jitter default is byte-identical to the pre-jitter code
# ---------------------------------------------------------------------------


class TestNoJitterRegressionGuard:
    """Every existing caller must keep byte-identical frames."""

    def _render(self, **kw):
        return render_frame(
            SENSOR, OPTICS, _defective_detector(),
            stars=[_bright_star(200.0, 150.0)],
            satellites=[SatelliteSpec(magnitude=10.0, angular_velocity_deg_s=0.3)],
            rng=np.random.default_rng(99), **kw,
        )

    def test_stage_is_first_and_removable_without_effect(self) -> None:
        """The new stage heads the chain; deleting it changes nothing when off."""
        assert default_pipeline()[0] is stage_pointing_jitter
        legacy_chain = default_pipeline()[1:]
        assert np.array_equal(
            self._render().data_float, self._render(pipeline=legacy_chain).data_float
        )

    def test_explicit_none_matches_default(self) -> None:
        assert np.array_equal(
            self._render().data_float, self._render(pointing_jitter=None).data_float
        )

    def test_zero_rms_matches_no_jitter(self) -> None:
        jittered = self._render(
            pointing_jitter=PointingJitter(rms_px=0.0, seed=5), frame_index=9
        )
        assert np.array_equal(self._render().data_float, jittered.data_float)

    def test_frame_index_ignored_without_a_model(self) -> None:
        assert np.array_equal(
            self._render(frame_index=0).data_float,
            self._render(frame_index=42).data_float,
        )

    def test_clean_path_unchanged(self) -> None:
        def clean(**kw):
            return generate_frame(
                SENSOR, OPTICS, stars=[_bright_star(200.0, 150.0)],
                rng=np.random.default_rng(4), **kw,
            ).data_float

        assert np.array_equal(clean(), clean(pointing_jitter=None, frame_index=6))
        assert np.array_equal(
            clean(), clean(pointing_jitter=PointingJitter(rms_px=0.0), frame_index=6)
        )


# ---------------------------------------------------------------------------
# (d) noise integrity — the second defect of the legacy corruptor
# ---------------------------------------------------------------------------


class TestNoiseIntegrity:
    """Scene jitter leaves the noise field alone; a post-render resample does not."""

    #: Source-free corner used as the noise probe (the star sits at 200, 150).
    PROBE = np.s_[0:80, 250:400]

    def _frames(self, pj: PointingJitter | None, n: int = 8):
        det = _clean_detector()
        return [
            render_frame(
                SENSOR, OPTICS, det,
                sky_mag_arcsec2=15.0,  # bright sky → shot-noise-dominated probe
                stars=[_bright_star(200.0, 150.0)],
                pointing_jitter=pj, frame_index=k,
                rng=np.random.default_rng(400 + k),
            ).data_float
            for k in range(n)
        ]

    @staticmethod
    def _probe_std(frames) -> float:
        return float(np.mean([f[TestNoiseIntegrity.PROBE].std() for f in frames]))

    def test_variance_matches_no_jitter(self) -> None:
        plain = self._probe_std(self._frames(None))
        jittered = self._probe_std(self._frames(PointingJitter(rms_px=0.5, seed=8)))
        assert jittered == pytest.approx(plain, rel=0.02)

    def test_post_render_resample_attenuates_variance(self) -> None:
        """Documents the legacy corruptor's optimistic bias (order-1 shift)."""
        plain_frames = self._frames(None)
        plain = self._probe_std(plain_frames)
        # A diagonal half-pixel shift averages four neighbours at equal weight,
        # so white-noise variance falls 3.97x (std 1.99x) — measured 3.93x on
        # this probe (recomputed 2026-07-27). A 1-D half-pixel shift costs 2x.
        resampled = [
            ndimage_shift(f, (0.5, 0.5), order=1, mode="nearest") for f in plain_frames
        ]
        assert self._probe_std(resampled) < 0.6 * plain
        # Even a modest 0.25 px shift measurably attenuates.
        quarter = [
            ndimage_shift(f, (0.25, 0.25), order=1, mode="nearest")
            for f in plain_frames
        ]
        assert self._probe_std(quarter) < 0.9 * plain

    def test_jitter_does_not_perturb_the_noise_rng(self) -> None:
        """Turning jitter on must not consume draws from the frame RNG."""
        det = _clean_detector()
        kw = dict(sky_mag_arcsec2=15.0, stars=[], satellites=[])
        plain = render_frame(
            SENSOR, OPTICS, det, rng=np.random.default_rng(77), **kw
        ).data_float
        jittered = render_frame(
            SENSOR, OPTICS, det, rng=np.random.default_rng(77),
            pointing_jitter=PointingJitter(rms_px=1.0, seed=2), frame_index=3, **kw,
        ).data_float
        # No sources ⇒ the offset has nothing to move, and the noise draw is
        # untouched, so the frames are bit-for-bit equal.
        assert np.array_equal(plain, jittered)
