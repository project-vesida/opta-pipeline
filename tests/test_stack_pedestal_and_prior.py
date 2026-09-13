"""Unit tests for the pedestal-safe stacked-image detection and the
ephemeris-vector velocity prior (findings #8/#9).

These are fast, array-level tests: no shift-and-add, no synthetic pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from opta_pipeline.calibrate import estimate_background
from opta_pipeline.config import PipelineConfig
from opta_pipeline.detect import detect_sources
from opta_pipeline.stack import Stacker


def _detect_on_stacked(stacked, noise_rms, cfg, star_mask):
    """Background-subtract a stacked image, then detect (the pedestal fix).

    Shift-and-add re-accumulates a background pedestal (per-frame residual
    summed ∝N), so ``detect_sources``' absolute threshold must be applied
    to the re-background-subtracted image.  This mirrors the demoted classic
    path's helper (now living in scripts/stacking_diagnostics.py); the
    likelihood path applies the same fix inline to its SNR map.
    """
    background, _ = estimate_background(
        stacked,
        box_size=cfg.calibration.background_box_size,
        filter_size=cfg.calibration.background_filter_size,
    )
    return detect_sources(
        stacked - background,
        noise_rms=noise_rms,
        snr_threshold=cfg.detection.snr_threshold,
        min_pixels=cfg.detection.min_streak_pixels,
        max_pixels=cfg.detection.max_streak_pixels,
        elongation_threshold=cfg.detection.elongation_threshold,
        star_mask=star_mask,
    )


def _stacked_with_pedestal(
    shape: tuple[int, int] = (200, 200),
    pedestal: float = 40.0,
    noise: float = 5.0,
    peak: float = 300.0,
    sigma_px: float = 2.0,
    seed: int = 0,
) -> np.ndarray:
    """A stacked-like image: large uniform DC pedestal + noise + one point source.

    Mimics the shift-and-add output whose per-frame background residual summed
    into a DC pedestal well above the ``noise·k`` threshold.
    """
    rng = np.random.default_rng(seed)
    img = rng.normal(0.0, noise, shape) + pedestal
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    cy, cx = shape[0] / 2.0, shape[1] / 2.0
    img += peak * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma_px**2)))
    return img


def test_pedestal_swamps_absolute_threshold_but_not_background_relative() -> None:
    """The raw absolute threshold loses the source in the pedestal; the
    background-subtracted path recovers it alone."""
    cfg = PipelineConfig.default()
    noise = 5.0
    img = _stacked_with_pedestal(noise=noise)
    cx = cy = img.shape[0] / 2.0

    # Raw detect_sources: threshold = noise·3 = 15 < pedestal 40, so ~every pixel
    # is above threshold — the giant blob is rejected by max_pixels, so the true
    # source is NOT recovered as a compact detection.
    raw = detect_sources(img, noise_rms=noise, snr_threshold=3.0, min_pixels=3)
    assert not any(abs(d.x - cx) < 3 and abs(d.y - cy) < 3 for d in raw)

    # Background-subtracted: the source is recovered as a compact detection.
    clean = _detect_on_stacked(img, noise, cfg, star_mask=None)
    assert any(abs(d.x - cx) < 3 and abs(d.y - cy) < 3 for d in clean)


def test_detect_on_stacked_clean_frame_is_noop() -> None:
    """On an already background-subtracted (≈zero-median) frame the fix is a
    no-op: the source is still found and no pedestal is invented."""
    cfg = PipelineConfig.default()
    noise = 5.0
    img = _stacked_with_pedestal(pedestal=0.0, noise=noise)
    cx = cy = img.shape[0] / 2.0
    clean = _detect_on_stacked(img, noise, cfg, star_mask=None)
    assert any(abs(d.x - cx) < 3 and abs(d.y - cy) < 3 for d in clean)


def test_from_prior_grid_is_centred_and_sized() -> None:
    """Stacker.from_prior brackets the predicted vector at the given step."""
    s = Stacker.from_prior(10.0, -20.0, half_width_px_s=2.0, step_px_s=1.0)
    assert s.vx_grid.min() == pytest.approx(8.0)
    assert s.vx_grid.max() == pytest.approx(12.0)
    assert s.vy_grid.min() == pytest.approx(-22.0)
    assert s.vy_grid.max() == pytest.approx(-18.0)
    assert s.vx_grid.size == 5 and s.vy_grid.size == 5


def test_from_prior_rejects_nonpositive_step() -> None:
    with pytest.raises(ValueError, match="step_px_s must be > 0"):
        Stacker.from_prior(0.0, 0.0, half_width_px_s=2.0, step_px_s=0.0)
