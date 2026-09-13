"""Velocity-aware star-mask veto (`_trajectory_dominated_by_star`).

Regression for the star-mask completeness ceiling (TODO.md;
docs/pipeline-stack-assessment.md F5): the stacked-detection veto used to drop
any candidate whose *reference-epoch* peak fell in the static catalog-star mask
— an SNR-independent ~13% loss for movers in star-dense pointings (a 15.2σ
prior candidate was vetoed).  The fix vetoes a candidate only when its whole
*trajectory* dwells on the mask (a static-star artefact), so a mover that merely
transits a masked pixel at its reference epoch survives while true static-star
rejection is preserved.
"""

from __future__ import annotations

import numpy as np

from opta_pipeline.pipeline import _trajectory_dominated_by_star


def _mask_with_star(shape: tuple[int, int], cx: int, cy: int, r: int) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


# Symmetric epochs about the reference (t=0), 40 frames at 25 fps.
_TIMES = [(k - 19.5) / 25.0 for k in range(40)]


def test_static_source_on_star_is_vetoed() -> None:
    """A v≈0 source sitting on a masked star dwells there the whole pass."""
    mask = _mask_with_star((300, 400), cx=200, cy=150, r=5)
    assert _trajectory_dominated_by_star(
        200.0, 150.0, 0.0, 0.0, _TIMES, mask, max_frac=0.5
    )


def test_mover_transiting_star_at_reference_epoch_survives() -> None:
    """Reference-epoch peak on the star, but a real mover only clips it."""
    mask = _mask_with_star((300, 400), cx=200, cy=150, r=5)
    # 200 px/s over ±0.78 s sweeps ±156 px — on the 10 px-wide star for only a
    # couple of frames near t=0.
    assert not _trajectory_dominated_by_star(
        200.0, 150.0, 200.0, 0.0, _TIMES, mask, max_frac=0.5
    )


def test_slow_mover_dominated_by_star_is_vetoed() -> None:
    """A hypothesis that crawls along the masked region is still rejected."""
    mask = _mask_with_star((300, 400), cx=200, cy=150, r=40)
    # 2 px/s over ±0.78 s stays within ±1.6 px of centre — inside the r=40 disc
    # for every frame.
    assert _trajectory_dominated_by_star(
        200.0, 150.0, 2.0, 0.0, _TIMES, mask, max_frac=0.5
    )


def test_no_mask_never_vetoes() -> None:
    empty = np.zeros((300, 400), dtype=bool)
    assert not _trajectory_dominated_by_star(
        200.0, 150.0, 0.0, 0.0, _TIMES, empty, max_frac=0.5
    )
    assert not _trajectory_dominated_by_star(
        200.0, 150.0, 0.0, 0.0, _TIMES, None, max_frac=0.5
    )


def test_fraction_threshold_boundary() -> None:
    """The veto triggers exactly when the on-mask fraction reaches max_frac."""
    mask = _mask_with_star((300, 400), cx=200, cy=150, r=3)
    times = [-2.0, -1.0, 0.0, 1.0]  # 4 epochs
    # vx = 3 px/s: positions x = 194, 197, 200, 203 -> within r=3 of x=200 for
    # t in {-1,0,1} (|dx| = 3,0,3) => 3/4 = 0.75 on-mask.
    assert _trajectory_dominated_by_star(200.0, 150.0, 3.0, 0.0, times, mask, 0.75)
    assert not _trajectory_dominated_by_star(200.0, 150.0, 3.0, 0.0, times, mask, 0.80)


def test_frac_zero_does_not_veto_a_trajectory_that_never_touches_the_mask() -> None:
    """max_frac=0.0 means "veto on any mask contact", not "veto everything".

    ``n_on >= 0 * n_frames`` holds with zero dwell, so the bare comparison
    vetoed every candidate in the frame — a total detection blackout from a
    config value that reads as the *loosest* possible setting.
    """
    mask = _mask_with_star((300, 400), cx=10, cy=10, r=5)
    # A trajectory in the far corner, nowhere near the star.
    assert not _trajectory_dominated_by_star(
        350.0, 250.0, 5.0, 0.0, _TIMES, mask, max_frac=0.0
    )


def test_frac_zero_vetoes_on_any_dwell() -> None:
    """The other half of the boundary: one on-mask frame is enough at 0.0."""
    mask = _mask_with_star((300, 400), cx=200, cy=150, r=3)
    # 50 px/s over ±0.78 s sweeps ±39 px: only the few epochs nearest t=0 land
    # on the r=3 disc — well under any sane max_frac, but non-zero dwell.
    assert not _trajectory_dominated_by_star(
        200.0, 150.0, 50.0, 0.0, _TIMES, mask, max_frac=0.5
    )
    assert _trajectory_dominated_by_star(
        200.0, 150.0, 50.0, 0.0, _TIMES, mask, max_frac=0.0
    )


def test_frac_zero_with_per_frame_masks_needs_dwell_too() -> None:
    """Same rule on the drift-tracking per-frame mask path."""
    masks = [_mask_with_star((300, 400), cx=10, cy=10, r=5) for _ in _TIMES]
    assert not _trajectory_dominated_by_star(
        350.0, 250.0, 5.0, 0.0, _TIMES, masks, max_frac=0.0
    )
