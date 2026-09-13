"""Tests for the realistic detector layer and effects-chain renderer.

Covers Phase 0 (sky-brightness convention) and Phase 1 (detector realism:
bias/FPN, flat/PRNU/vignetting, dark-current map with hot/dead pixels,
full-well saturation) plus calibration-frame generation and the
calibrate-recovers-truth contract.

Uses SENSOR_SMALL (400×300) for speed; physics still derives from
opta_model.radiometry (AGENTS.md rule #4).
"""

from __future__ import annotations

import numpy as np
import pytest
from opta_model.radiometry import sky_background_electrons
from sensor_fixtures import OPTICS_DEFAULT, SENSOR_SMALL

from opta_pipeline.calibrate import (
    calibrate_frame,
    make_master_dark,
    make_master_flat,
)
from opta_pipeline.synth import (
    StarSpec,
    build_detector_model,
    generate_dark_frame,
    generate_flat_frame,
    generate_frame,
    sky_electrons_per_pixel,
)
from opta_pipeline.synth.render import RenderContext, default_pipeline, render_frame

SENSOR = SENSOR_SMALL
OPTICS = OPTICS_DEFAULT


# ---------------------------------------------------------------------------
# Phase 0 — sky-brightness convention
# ---------------------------------------------------------------------------


class TestSkyConvention:
    """The explicit observed/above-atmosphere convention switch."""

    def _args(self):
        from opta_model.hardware import compute_pixel_scale

        ps = compute_pixel_scale(SENSOR.pixel_size_um, OPTICS.focal_length_mm)
        return dict(
            sky_mag_arcsec2=21.0,
            pixel_scale_arcsec=ps,
            aperture_m=OPTICS.aperture_mm / 1000.0,
            integration_time_s=1.0 / SENSOR.frame_rate_hz,
            quantum_efficiency=SENSOR.quantum_efficiency,
        )

    def test_observed_matches_radiometry(self) -> None:
        """sky_is_observed=True must match the analytic model exactly."""
        a = self._args()
        observed = sky_electrons_per_pixel(
            **a, elevation_deg=45.0, sky_is_observed=True
        )
        analytic = sky_background_electrons(
            a["sky_mag_arcsec2"],
            a["pixel_scale_arcsec"],
            a["aperture_m"],
            a["integration_time_s"],
            a["quantum_efficiency"],
        )
        assert observed == pytest.approx(analytic, rel=1e-12)

    def test_default_extincts_and_is_dimmer(self) -> None:
        """Default (above-atmosphere) applies extinction → fewer sky electrons."""
        a = self._args()
        default = sky_electrons_per_pixel(**a, elevation_deg=45.0)
        observed = sky_electrons_per_pixel(
            **a, elevation_deg=45.0, sky_is_observed=True
        )
        assert default < observed  # extinction attenuates the modelled sky

    def test_realistic_path_uses_observed_convention(self) -> None:
        """render_frame's sky must equal the observed-convention value."""
        det = build_detector_model(SENSOR, seed=1)
        a = self._args()
        frame = generate_frame(
            SENSOR, OPTICS, sky_mag_arcsec2=21.0, detector=det,
            rng=np.random.default_rng(0),
        )
        observed = sky_background_electrons(
            21.0, a["pixel_scale_arcsec"], a["aperture_m"],
            a["integration_time_s"], a["quantum_efficiency"],
        )
        assert frame.sky_e_per_pixel == pytest.approx(observed, rel=1e-9)


# ---------------------------------------------------------------------------
# Phase 1 — detector model
# ---------------------------------------------------------------------------


class TestDetectorModel:
    """build_detector_model produces deterministic, frame-stable structure."""

    def test_shape_and_flat_centred(self) -> None:
        det = build_detector_model(SENSOR, seed=3)
        assert det.shape == (SENSOR.resolution_v, SENSOR.resolution_h)
        assert det.flat.shape == det.shape
        # Mean response near unity (vignetting pulls it slightly below 1).
        assert 0.85 < float(det.flat.mean()) < 1.01

    def test_deterministic_for_seed(self) -> None:
        d1 = build_detector_model(SENSOR, seed=5)
        d2 = build_detector_model(SENSOR, seed=5)
        assert np.array_equal(d1.flat, d2.flat)
        assert np.array_equal(d1.bias_fpn_e, d2.bias_fpn_e)
        assert np.array_equal(d1.dark_current_e_s, d2.dark_current_e_s)

    def test_seed_changes_pattern(self) -> None:
        d1 = build_detector_model(SENSOR, seed=5)
        d2 = build_detector_model(SENSOR, seed=6)
        assert not np.array_equal(d1.flat, d2.flat)

    def test_hot_and_dead_pixels_present(self) -> None:
        det = build_detector_model(
            SENSOR, seed=7, hot_pixel_fraction=1e-3, dead_pixel_fraction=1e-3,
            hot_pixel_dark_e_s=50.0,
        )
        assert det.dark_current_e_s.max() > 10.0  # hot pixels
        assert det.flat.min() < 0.1  # dead pixels collapse the response

    def test_vignetting_falls_off_to_corner(self) -> None:
        det = build_detector_model(
            SENSOR, seed=8, prnu_pct=0.0, vignetting_corner_factor=0.7,
            hot_pixel_fraction=0.0, dead_pixel_fraction=0.0,
        )
        h, w = det.shape
        centre = float(det.flat[h // 2, w // 2])
        corner = float(det.flat[0, 0])
        assert corner < centre
        assert corner == pytest.approx(0.7, abs=0.05)


# ---------------------------------------------------------------------------
# Phase 1 — realistic rendering path
# ---------------------------------------------------------------------------


class TestRealisticPath:
    """generate_frame imprints the instrumental signature when given a model."""

    def test_clean_path_unchanged_without_detector(self) -> None:
        """detector=None stays deterministic and uint16 (legacy contract)."""
        f1 = generate_frame(SENSOR, OPTICS, rng=np.random.default_rng(0))
        f2 = generate_frame(SENSOR, OPTICS, rng=np.random.default_rng(0))
        assert f1.data.dtype == np.uint16
        assert np.array_equal(f1.data, f2.data)

    def test_bias_pedestal_in_background(self) -> None:
        det = build_detector_model(
            SENSOR, seed=9, bias_offset_e=300.0, hot_pixel_fraction=0.0
        )
        frame = generate_frame(
            SENSOR, OPTICS, sky_mag_arcsec2=21.0, detector=det,
            rng=np.random.default_rng(0),
        )
        # Background sits near the bias pedestal (+ small sky/dark).
        assert 295.0 < float(np.median(frame.data_float)) < 320.0

    def test_saturation_clips_to_full_well(self) -> None:
        det = build_detector_model(SENSOR, seed=10)
        # A very bright star drives the core past full well.
        frame = generate_frame(
            SENSOR, OPTICS, stars=[StarSpec(magnitude=-2.0, x=200, y=150)],
            detector=det, rng=np.random.default_rng(0),
        )
        assert float(frame.data_float.max()) <= det.full_well_e + 1.0
        assert float(frame.data_float.max()) > 0.9 * det.full_well_e

    def test_sources_carry_shot_noise(self) -> None:
        """Realistic path applies Poisson to the whole collected charge."""
        det = build_detector_model(
            SENSOR, seed=11, prnu_pct=0.0, bias_fpn_rms_e=0.0,
            hot_pixel_fraction=0.0, dead_pixel_fraction=0.0,
        )
        # Two frames, different RNG → bright star core differs (shot noise).
        kw = dict(stars=[StarSpec(magnitude=8.0, x=200, y=150)], detector=det)
        a = generate_frame(SENSOR, OPTICS, **kw, rng=np.random.default_rng(1))
        b = generate_frame(SENSOR, OPTICS, **kw, rng=np.random.default_rng(2))
        assert not np.array_equal(a.data_float, b.data_float)


# ---------------------------------------------------------------------------
# Phase 1 — calibration frames + recover-truth
# ---------------------------------------------------------------------------


class TestCalibrationFrames:
    """Dark/flat generation + the calibrate-recovers-truth contract."""

    def test_dark_and_flat_headers(self) -> None:
        det = build_detector_model(SENSOR, seed=12)
        dk = generate_dark_frame(SENSOR, det, rng=np.random.default_rng(0))
        fl = generate_flat_frame(SENSOR, det, rng=np.random.default_rng(0))
        assert dk.header["IMAGETYP"] == "DARK"
        assert fl.header["IMAGETYP"] == "FLAT"

    def test_default_rng_calibration_frames_are_independent_draws(self) -> None:
        """``rng=None`` must NOT reuse render_frame's seeded-0 default.

        A master is built by sigma-clipping several frames; N bit-identical
        inputs survive the clip untouched and bake a single read/shot-noise
        realization into the master as false fixed-pattern structure, which
        ``calibrate_frame`` then subtracts from every science frame.
        """
        det = build_detector_model(SENSOR, seed=15, bias_offset_e=300.0)
        d1 = generate_dark_frame(SENSOR, det)
        d2 = generate_dark_frame(SENSOR, det)
        assert not np.array_equal(d1.data, d2.data)

        f1 = generate_flat_frame(SENSOR, det, illumination_e=8000.0)
        f2 = generate_flat_frame(SENSOR, det, illumination_e=8000.0)
        assert not np.array_equal(f1.data, f2.data)

    def test_explicit_rng_stays_reproducible(self) -> None:
        """The entropy default must not cost the explicit-seed contract."""
        det = build_detector_model(SENSOR, seed=15, bias_offset_e=300.0)
        for gen, kw in (
            (generate_dark_frame, {}),
            (generate_flat_frame, {"illumination_e": 8000.0}),
        ):
            a = gen(SENSOR, det, rng=np.random.default_rng(7), **kw)
            b = gen(SENSOR, det, rng=np.random.default_rng(7), **kw)
            np.testing.assert_array_equal(a.data, b.data)

    def test_dark_has_no_sky_signal(self) -> None:
        det = build_detector_model(SENSOR, seed=13, bias_offset_e=300.0,
                                   hot_pixel_fraction=0.0)
        dk = generate_dark_frame(SENSOR, det, rng=np.random.default_rng(0))
        # Dark ≈ bias + tiny dark current; no sky photons.
        assert float(np.median(dk.data_float)) == pytest.approx(300.0, abs=5.0)

    def test_calibration_recovers_flat_background_and_star(self) -> None:
        det = build_detector_model(
            SENSOR, seed=14, prnu_pct=3.0, vignetting_corner_factor=0.7,
            bias_offset_e=300.0, bias_fpn_rms_e=5.0, hot_pixel_fraction=1e-3,
        )
        darks = [
            generate_dark_frame(SENSOR, det, rng=np.random.default_rng(100 + i)).data
            for i in range(7)
        ]
        flats = [
            # 8000 e⁻ < full well (17000) so the flat is NOT saturated — a
            # saturated flat would clip away the vignetting and silently become
            # a no-op (see test_master_flat_recovers_response).
            generate_flat_frame(
                SENSOR, det, illumination_e=8000.0, rng=np.random.default_rng(200 + i)
            ).data
            for i in range(7)
        ]
        master_dark = make_master_dark(darks)
        master_flat = make_master_flat(flats)

        sci = generate_frame(
            SENSOR, OPTICS, sky_mag_arcsec2=21.0,
            stars=[StarSpec(magnitude=7.0, x=200, y=150)],
            detector=det, rng=np.random.default_rng(42),
        )
        raw = sci.data.astype(float)
        cal = calibrate_frame(
            sci.data, master_dark=master_dark, master_flat=master_flat,
            subtract_background=True,
        )

        mask = np.ones_like(cal.data, dtype=bool)
        mask[142:159, 192:209] = False  # exclude the star

        # Calibration removes the bias pedestal and flattens the structure.
        assert abs(float(cal.data[mask].mean())) < 2.0
        assert float(cal.data[mask].std()) < 0.5 * float(raw[mask].std())
        # The star survives and is well above the residual noise.
        star_peak = float(cal.data[145:156, 195:206].max())
        assert star_peak / max(cal.background_rms, 1e-6) > 20.0


# ---------------------------------------------------------------------------
# Architecture — pipeline extensibility
# ---------------------------------------------------------------------------


class TestPipelineExtensibility:
    """The effects chain is a plain editable list of stages."""

    def test_default_pipeline_is_editable_list(self) -> None:
        p = default_pipeline()
        assert isinstance(p, list)
        assert all(callable(s) for s in p)

    def test_custom_stage_runs(self) -> None:
        """A user-supplied stage is executed in order."""
        det = build_detector_model(SENSOR, seed=15, bias_offset_e=0.0,
                                   bias_fpn_rms_e=0.0, hot_pixel_fraction=0.0)
        marker = {"ran": False}

        def my_stage(ctx: RenderContext) -> None:
            marker["ran"] = True
            ctx.image_e += 500.0  # inject a constant pedestal

        pipeline = default_pipeline()
        pipeline.insert(0, my_stage)
        frame = render_frame(
            SENSOR, OPTICS, det, sky_mag_arcsec2=21.0, pipeline=pipeline,
            rng=np.random.default_rng(0),
        )
        assert marker["ran"] is True
        # The 500 e⁻ pedestal dominates the otherwise ~1 e⁻ sky background
        # (it is scaled by the flat field, hence < 500).
        assert float(np.median(frame.data_float)) > 400.0


# ---------------------------------------------------------------------------
# Contracts the future generator work depends on
# ---------------------------------------------------------------------------


class TestRenderContracts:
    """Reproducibility, saturation/nonlinearity, and master-frame fidelity.

    These guard properties that later phases (oversampling, scene stressors,
    photometry) will build on — without them, regressions would surface as
    confusing downstream failures rather than a precise local one.
    """

    def test_realistic_path_is_reproducible(self) -> None:
        """Same detector + same RNG seed → byte-identical frame (both arrays)."""
        det = build_detector_model(SENSOR, seed=20)
        kw = dict(
            stars=[StarSpec(magnitude=9.0, x=200, y=150)], detector=det,
            sky_mag_arcsec2=21.0,
        )
        a = generate_frame(SENSOR, OPTICS, **kw, rng=np.random.default_rng(5))
        b = generate_frame(SENSOR, OPTICS, **kw, rng=np.random.default_rng(5))
        assert np.array_equal(a.data, b.data)
        assert np.array_equal(a.data_float, b.data_float)

    def test_different_seed_changes_noise(self) -> None:
        det = build_detector_model(SENSOR, seed=20)
        kw = dict(detector=det, sky_mag_arcsec2=21.0)
        a = generate_frame(SENSOR, OPTICS, **kw, rng=np.random.default_rng(5))
        b = generate_frame(SENSOR, OPTICS, **kw, rng=np.random.default_rng(6))
        assert not np.array_equal(a.data_float, b.data_float)

    def test_nonlinearity_compresses_near_full_well(self) -> None:
        """A non-zero nonlinearity coefficient suppresses high-signal pixels.

        Exercises the ``k > 0`` branch of stage_saturate, which is otherwise
        dead code.  With a flat (=1) response and illumination at 0.8·FW, the
        compression factor is ``1 − k·0.8``.
        """
        fw = SENSOR.full_well_e
        common = dict(
            prnu_pct=0.0, bias_fpn_rms_e=0.0, bias_offset_e=0.0,
            hot_pixel_fraction=0.0, dead_pixel_fraction=0.0,
            vignetting_corner_factor=1.0,
        )
        linear = build_detector_model(SENSOR, seed=21, nonlinearity=0.0, **common)
        compressed = build_detector_model(SENSOR, seed=21, nonlinearity=0.3, **common)
        f_lin = generate_flat_frame(
            SENSOR, linear, illumination_e=0.8 * fw, rng=np.random.default_rng(0)
        )
        f_nl = generate_flat_frame(
            SENSOR, compressed, illumination_e=0.8 * fw, rng=np.random.default_rng(0)
        )
        ratio = float(np.median(f_nl.data_float) / np.median(f_lin.data_float))
        assert ratio == pytest.approx(1.0 - 0.3 * 0.8, abs=0.02)

    def test_master_flat_recovers_response(self) -> None:
        """make_master_flat reconstructs the detector's vignetting/PRNU.

        Guards against the saturated-flat failure mode: an over-illuminated
        flat clips to full well and yields master_flat ≈ 1 (no correction).
        """
        det = build_detector_model(
            SENSOR, seed=22, prnu_pct=1.0, vignetting_corner_factor=0.7,
            hot_pixel_fraction=0.0, dead_pixel_fraction=0.0,
        )
        flats = [
            generate_flat_frame(
                SENSOR, det, illumination_e=8000.0, rng=np.random.default_rng(300 + i)
            ).data
            for i in range(9)
        ]
        master_flat = make_master_flat(flats)
        ref = det.flat / np.median(det.flat)
        h, w = det.shape
        master_ratio = float(master_flat[0, 0] / master_flat[h // 2, w // 2])
        true_ratio = float(ref[0, 0] / ref[h // 2, w // 2])
        assert master_ratio == pytest.approx(true_ratio, abs=0.05)
        assert true_ratio < 0.8  # vignetting really is present (not a no-op)

    def test_saturated_flat_loses_structure(self) -> None:
        """Document the failure mode: an over-illuminated flat is a no-op.

        Pins the rationale for the safe default illumination — if a caller
        floods the flat past full well, the vignetting is clipped away.
        """
        det = build_detector_model(
            SENSOR, seed=22, prnu_pct=0.0, vignetting_corner_factor=0.7,
            hot_pixel_fraction=0.0, dead_pixel_fraction=0.0,
        )
        flats = [
            generate_flat_frame(
                SENSOR, det, illumination_e=5.0 * det.full_well_e,
                rng=np.random.default_rng(300 + i),
            ).data
            for i in range(5)
        ]
        master_flat = make_master_flat(flats)
        h, w = det.shape
        # Saturated → uniform → corner/center ≈ 1 (structure lost).
        assert master_flat[0, 0] / master_flat[h // 2, w // 2] == pytest.approx(
            1.0, abs=0.02
        )

    def test_master_dark_recovers_bias_and_dark(self) -> None:
        """make_master_dark reconstructs the bias pedestal + dark + hot pixels."""
        det = build_detector_model(
            SENSOR, seed=23, bias_offset_e=300.0, bias_fpn_rms_e=4.0,
            hot_pixel_fraction=1e-3, hot_pixel_dark_e_s=50.0,
        )
        darks = [
            generate_dark_frame(SENSOR, det, rng=np.random.default_rng(400 + i)).data
            for i in range(9)
        ]
        master_dark = make_master_dark(darks)
        t = 1.0 / SENSOR.frame_rate_hz
        expected = 300.0 + SENSOR.dark_current_e_s * t  # gain = 1
        assert float(np.median(master_dark)) == pytest.approx(expected, abs=1.0)
        # Hot pixels (stable) survive the median stack and are captured.
        assert float(master_dark.max()) > 10.0
