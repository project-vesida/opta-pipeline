"""Pipeline configuration — loads pipeline_defaults.yaml into frozen dataclasses.

Single source of truth for all pipeline tuning parameters. Every module imports
its defaults from here; no magic numbers in pipeline code (AGENTS.md rule #3).

Usage
-----
    from opta_pipeline.config import PipelineConfig

    cfg = PipelineConfig.default()           # loads pipeline_defaults.yaml
    cfg = PipelineConfig.from_yaml(path)     # loads a custom override file

To override individual parameters without touching the YAML:

    from dataclasses import replace
    cfg = replace(PipelineConfig.default(),
                  tracklet=replace(cfg.tracklet, linear_fit_residual_arcsec=50.0))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

DEFAULTS_PATH: Path = Path(
    str(files("opta_pipeline") / "configs" / "pipeline_defaults.yaml")
)

__all__ = [
    "DEFAULTS_PATH",
    "DEFAULT_HARDWARE_PROFILE",
    "DESIGN_MAX_ANGULAR_RATE_DEG_S",
    "MAX_SEP_RATE_HEADROOM",
    "RESIDUAL_PRUNE_SIGMA",
    "CENTROIDING_FRACTION_PX",
    "DISTORTION_RESIDUAL_ARCSEC",
    "derive_max_sep_arcsec",
    "derive_max_residual_arcsec",
    "CalibrationConfig",
    "DetectionConfig",
    "AstrometryConfig",
    "RollingShutterConfig",
    "TrackletConfig",
    "StackingConfig",
    "PipelineConfig",
]


# ---------------------------------------------------------------------------
# Per-optics gate derivation
# ---------------------------------------------------------------------------
#
# The two scale/rate-sensitive pipeline gates (the tracklet association gate
# and the astrometric per-star outlier prune) DERIVE from the hardware profile
# the config declares (``hardware.profile`` in pipeline_defaults.yaml) instead
# of being absolute constants.  Absolute constants silently break when the
# optics change: the historical ``max_sep_arcsec = 300`` was a hard
# ~2.08 deg/s rate ceiling (and only ~1.9 px/frame at all-sky plate scales —
# meteor-blind), and the historical ``max_residual_arcsec = 4.0`` was 0.17 px
# at the selected v3 node's 23.93 arcsec/px — a sub-pixel-impossible per-star
# prune that could never legitimately fire.

#: Hardware profile the shipped defaults resolve gates against: the T-01 v3
#: selected node (7Artisans 25 mm f/0.95 + IMX585) in the 25 fps 1920x1080
#: ROI readout baseline — the same node ``opta_validation`` runs as
#: ``validation_operational`` (configs/hardware_catalog/profiles.yaml).
DEFAULT_HARDWARE_PROFILE: str = "selected_v3_roi"

#: Design maximum angular rate (deg/s) the node is sized to track.  Source:
#: T-03 (SYSTEMS.md decision log 2026-05-27) sizes the mandatory
#: rolling-shutter correction to 2.0 deg/s (corrected residual <= 1.4 arcsec
#: at 2.0 deg/s) — the fastest LEO rate in the system design.  Not a tuned
#: number; change it only with a SYSTEMS.md decision-log entry.
DESIGN_MAX_ANGULAR_RATE_DEG_S: float = 2.0

#: Stated headroom factor on the design rate for the association gate.  1.25
#: places the acquisition ceiling exactly at the ratified Level-A correlation
#: rate bound (2.5 deg/s, ``opta_validation.criteria.OPTA_COMMISSIONING``
#: ``max_angular_velocity_deg_s``; PO-1 criterion, SYSTEMS.md 2026-07-13):
#: the linker can acquire anything the ratified correlation criterion would
#: accept, and nothing faster.
MAX_SEP_RATE_HEADROOM: float = 1.25

#: The astrometric per-star prune rejects at this many sigma of the expected
#: per-star WCS residual (Gaussian 3-sigma outlier rejection: keeps 99.7 % of
#: genuine stars).
RESIDUAL_PRUNE_SIGMA: float = 3.0


#: Per-star centroiding uncertainty as a fraction of the pixel scale
#: (Gaussian fitting on a well-sampled star).  Same default as
#: ``opta_model.error_budget.astrometric_error_budget``; kept here as a plain
#: number so the pipeline runtime does not depend on opta-model.
CENTROIDING_FRACTION_PX: float = 0.3

#: Residual distortion (arcsec RMS) after polynomial/SIP calibration, the
#: second per-star term of the OpTA.NOD.ACC error budget (SYSTEMS.md).  Same
#: default as ``opta_model.error_budget.astrometric_error_budget``.
DISTORTION_RESIDUAL_ARCSEC: float = 2.0


def _profile_numbers(profile: str) -> tuple[float, float]:
    """``(frame_rate_hz, pixel_scale_arcsec)`` for a hardware-catalog profile.

    Resolving a profile *name* needs the optional ``opta-model`` dependency
    (``pip install 'opta-pipeline[synth]'``).  A runtime config that states
    ``hardware.frame_rate_hz`` and ``hardware.pixel_scale_arcsec`` directly
    never takes this path.
    """
    try:
        from opta_model.hardware_catalog import build_node_from_profile
    except ImportError as exc:  # pragma: no cover - exercised without opta-model
        raise ImportError(
            f"resolving hardware profile {profile!r} needs opta-model "
            "(pip install 'opta-pipeline[synth]'); alternatively set "
            "hardware.frame_rate_hz and hardware.pixel_scale_arcsec in the "
            "pipeline config so the gates derive without a catalog lookup"
        ) from exc
    node = build_node_from_profile(profile)
    return float(node.sensor.frame_rate_hz), float(node.pixel_scale_arcsec)


@cache
def derive_max_sep_arcsec(
    hardware: str | float = DEFAULT_HARDWARE_PROFILE,
    design_rate_deg_s: float = DESIGN_MAX_ANGULAR_RATE_DEG_S,
    headroom: float = MAX_SEP_RATE_HEADROOM,
) -> float:
    """Tracklet association gate (arcsec per elapsed frame).

    ``design_rate_deg_s x 3600 / fps x headroom``: the sky step one frame
    period of motion at the design maximum angular rate produces, times the
    stated headroom factor (see :data:`MAX_SEP_RATE_HEADROOM`).

    ``hardware`` is either the frame rate in Hz or a hardware-catalog profile
    name (needs opta-model).  At 25 fps: 2.0 * 3600 / 25 * 1.25 = 360.0.
    """
    fps = (
        _profile_numbers(hardware)[0] if isinstance(hardware, str) else float(hardware)
    )
    return design_rate_deg_s * 3600.0 / fps * headroom


@cache
def derive_max_residual_arcsec(
    hardware: str | float = DEFAULT_HARDWARE_PROFILE,
    prune_sigma: float = RESIDUAL_PRUNE_SIGMA,
) -> float:
    """Astrometric per-star outlier prune (arcsec).

    ``prune_sigma x RSS(centroiding_fraction x plate_scale, distortion)``,
    the per-star terms of the WCS error budget
    (:data:`CENTROIDING_FRACTION_PX`, :data:`DISTORTION_RESIDUAL_ARCSEC`;
    the same terms ``opta_model.error_budget.astrometric_error_budget`` sums
    to close the 10 arcsec OpTA.NOD.ACC requirement).  A star whose residual
    exceeds ``prune_sigma`` times the expected per-star scatter is a mismatch
    or a bad centroid, not astrometric noise.

    ``hardware`` is either the pixel scale in arcsec/px or a hardware-catalog
    profile name (needs opta-model).  At 23.927 arcsec/px: 3 x 7.451 = 22.354.
    """
    scale = (
        _profile_numbers(hardware)[1] if isinstance(hardware, str) else float(hardware)
    )
    per_star = math.hypot(
        CENTROIDING_FRACTION_PX * scale, DISTORTION_RESIDUAL_ARCSEC
    )
    return prune_sigma * per_star


def _gate(raw: Any, deriver: Any, hardware: str | float) -> float:
    """Resolve a YAML gate value: the sentinel ``"derived"`` or a number."""
    if raw == "derived":
        return float(deriver(hardware))
    return float(raw)


@dataclass(frozen=True)
class CalibrationConfig:
    """Mesh-background estimation parameters (master frames arrive via FrameContext)."""

    background_box_size: int
    background_filter_size: int


@dataclass(frozen=True)
class DetectionConfig:
    """Source-extraction thresholds and streak/point classification gates."""

    snr_threshold: float
    min_streak_pixels: int
    max_streak_pixels: int
    elongation_threshold: float
    mask_star_radius_px: int
    # Velocity-aware star veto (stacked path).  A stacked candidate is vetoed
    # by the static catalog-star mask only when its *trajectory* dwells on
    # masked pixels for at least this fraction of the pass — the signature of a
    # static source (v≈0), not of a mover that merely transits a star at its
    # reference epoch.  Vetoing on the reference-epoch pixel alone (frac→0)
    # discarded good movers SNR-independently (~13% loss in star-dense
    # pointings; see docs/pipeline-stack-assessment.md F5 / star-mask ceiling).
    # Must lie in [0, 1]: it is a fraction of the pass.  0.0 vetoes on any mask
    # contact at all, 1.0 only on full-pass dwell; > 1.0 is unreachable and
    # silently disabled the veto before this check existed.
    mask_veto_max_trajectory_frac: float = 0.5

    def __post_init__(self) -> None:
        frac = self.mask_veto_max_trajectory_frac
        if not 0.0 <= frac <= 1.0:
            raise ValueError(
                "mask_veto_max_trajectory_frac must be in [0, 1] "
                f"(fraction of the pass), got {frac}"
            )


@dataclass(frozen=True)
class AstrometryConfig:
    """Plate-solve outlier rejection, distortion model, and RS enable flag."""

    max_residual_arcsec: float
    rolling_shutter_correction: bool
    # T-11: fit a radial SIP distortion term (fit_wcs_sip) instead of a plain
    # linear TAN.  Mandatory for the wide, fast prime whose several-percent
    # barrel a linear solve cannot absorb; degrades to linear when distortion
    # is absent, so it is safe to leave on.
    sip_distortion: bool = True
    # Consumption policy for the post-solve OpTA.NOD.ACC self-check
    # (WCSSolution.accuracy_flag): False (default) = annotate-only — flagged
    # frames still contribute astrometry, but FrameDetections carry
    # wcs_accuracy_flag and tracklets containing them are annotated
    # (Tracklet.wcs_accuracy_flagged) for downstream QC.  True = flagged
    # frames are excluded from astrometry (treated like an unsolved frame).
    # Default OFF so first light collects data rather than dropping it;
    # detection (stacking runs in pixel space) is unaffected either way.
    skip_flagged_frames: bool = False


@dataclass(frozen=True)
class RollingShutterConfig:
    """Rolling-shutter timing correction parameters (T-03)."""

    enabled: bool
    row_readout_us: float
    reference_row: float


@dataclass(frozen=True)
class TrackletConfig:
    """Hungarian linker and linear-motion QC thresholds."""

    min_detections: int
    max_gap_frames: int
    linear_fit_residual_arcsec: float
    # Hungarian linker association gate: maximum sky separation (arcsec)
    # between a track's last matched point and a candidate detection, *per
    # elapsed frame* (the linker scales this by the frame-id gap, so gap
    # tolerance does not lower the maximum trackable rate).  In
    # pipeline_defaults.yaml this is the sentinel ``derived`` — resolved per
    # hardware profile by :func:`derive_max_sep_arcsec` (design max rate
    # 2.0 deg/s x 1.25 headroom -> 360 arcsec/frame at the 25 fps v3 ROI
    # baseline, an exact 2.5 deg/s rate ceiling).  The 300.0 here is only the
    # legacy API default for direct construction (~2.08 deg/s at 25 fps).
    max_sep_arcsec: float = 300.0


@dataclass(frozen=True)
class StackingConfig:
    """Blind track-and-stack velocity search grid (px/s) and scorer knobs."""

    enabled: bool
    velocity_max_px_s: float
    velocity_step_px_s: float
    # Minimum search speed (px/s): velocity hypotheses with |v| below this are
    # excluded from the stack search.  Static sources (residual stars, hot
    # pixels) only stack coherently near v=0, so excluding the near-zero region
    # suppresses their false peaks while letting a genuine mover win.  Default
    # 0.0 disables exclusion (legacy behaviour).
    velocity_min_px_s: float = 0.0
    # Long-pass windowing: split a sequence longer than one window into
    # locally-linear windows, each stacked independently.  ``window_duration_s
    # <= 0`` means "single window" (legacy whole-sequence behaviour).
    window_duration_s: float = 0.0
    window_overlap_s: float = 0.0
    # Cross-window duplicate-identity radius (arcsec) for overlapping
    # stacking windows (``run_windowed_track_and_stack``).  Two window
    # tracklets are the *same object* only if their fitted sky tracks agree
    # within astrometric error everywhere on the shared time span — a far
    # tighter statement than the linker's rate-derived *association* gate
    # (``TrackletConfig.max_sep_arcsec``, 360 arcsec/frame at the v3 ROI
    # baseline), which is sized for
    # inter-frame motion headroom and would silently merge distinct
    # co-moving neighbours (deployment clusters / satellite trains at
    # a-few-hundred-arcsec spacing).  Default 30 arcsec = 3 x the
    # OpTA.NOD.ACC 10 arcsec RMS tracklet-accuracy budget (SYSTEMS.md):
    # two independent window fits of one object each err <= ~10 arcsec RMS,
    # so their track-vs-track disagreement is ~sqrt(2)*10 ~= 14 arcsec RMS,
    # and 30 arcsec covers that at >2 sigma while staying 12 x below the
    # association gate (a 250 arcsec co-moving pair survives with >8 x
    # margin).
    dedup_radius_arcsec: float = 30.0
    # Capture rate (fps) assumed when it cannot be inferred from frame
    # timestamps (fewer than two distinct times).  Matches the IMX585 ROI
    # baseline frame rate (25 fps).
    fallback_fps: float = 25.0
    # Gaussian PSF sigma (px) for the likelihood matched-filter kernel.  MUST
    # match the synthetic-frame PSF the filter runs against: opta_pipeline.synth
    # renders at psf_fwhm_px = 2.0, so sigma = 2.0 / 2.355 ≈ 0.85 px
    # (likelihood.DEFAULT_PSF_SIGMA_PX).  The former 1.5 px was a sigma↔FWHM
    # confusion — a ~2× too-wide filter that forfeited ~14% matched-filter SNR.
    psf_sigma_px: float = 0.849
    # Safety margin (sigma) added on top of the Poisson-aware detection
    # threshold u* = poisson_tail_threshold(N_pix*N_hyp, eps) on the stacked
    # SNR map (Bernstein bound, contract since 2026-07-28; the former
    # Gaussian sqrt(2 ln(N_pix N_hyp)) is its eps=0 limit).  The bound
    # scale eps is estimated per batch from the frames/noise, never
    # configured — this knob remains pure extra margin.
    far_margin_sigma: float = 0.5
    # Blind-search architecture (Phase 2): windowed coarse seeding +
    # per-candidate pyramid refinement to the full-pass criterion step.
    # False = single-stage grid at velocity_step_px_s (criterion-violating
    # for long passes — see assessment F1; fine for small cued boxes).
    coarse_to_fine: bool = True
    # Stage-1 per-window seeding threshold (sigma).  Sets the blind-mode
    # sensitivity rolloff: completeness is probabilistic between u* and
    # roughly this value times sqrt(N/N_window).
    coarse_prethreshold_sigma: float = 4.0
    # Cap on candidates passed from stage-1 seeding to pyramid refinement.
    max_fine_candidates: int = 20
    # Stage-1 seeding window (seconds).  0 = auto: max(PSF_fwhm /
    # velocity_step_px_s, 4 frames, T*(prethreshold/u*)^2) — the last term
    # (added 2026-07-28) matches seeding depth to the final u* gate by
    # construction (seeding_floor(T, T0) <= u*(T)), so auto depth no longer
    # truncates long passes.  Longer windows seed deeper (per-window SNR
    # grows as sqrt(n)) at higher stage-1 cost (finer stage-1 grid step).
    seed_window_s: float = 0.0
    # Stage-1 seeding on b x b sum-binned psi/phi maps (1 = full resolution).
    # b=2 measured ~3x faster seeding; the binned SNR is renormalized by the
    # exact kernel-autocorrelation factor so prethreshold keeps its meaning.
    seed_binning: int = 1
    # F8 forced-photometry centroid-refinement gate (sigma).  A per-frame
    # measured centroid only replaces the predicted track position when the
    # local matched-filter peak clears this; below it the stack's velocity
    # solution is passed through (honest forced photometry, no fabricated
    # astrometry).  Deliberately DECOUPLED from detection.snr_threshold: at
    # ~3 sigma the max over the refinement search disc is dominated by noise,
    # so refinement snaps marginal targets onto noise peaks and inflates the
    # track's O-C RMS until the linker QC rejects a solidly stack-detected
    # object (e2e findings #11/#12).  The nominal gate is ~5 sigma (centroid
    # error psf_sigma/SNR ~ 0.3 px, the level where measurement starts to
    # beat the stack prediction); the default carries a ~1.4x margin because
    # the scalar per-frame noise_rms underestimates local photon noise near
    # bright-star pixels, which otherwise admits noise centroids exactly
    # where the SNR map is least trustworthy (finding #12: e2e O-C RMS
    # plateaus at the stack-solution floor for gates >= 7, but is still
    # noise-inflated at 5-6).
    refine_snr_min: float = 7.0
    # F4 within-exposure streak kernel (opt-in; default off).  A source
    # moving at |v| px/s trails |v|*exposure_time_s px within one frame, so
    # the round-PSF matched filter loses SNR.  When enabled, the winning
    # velocity (blind final gate, prior/single-stage winner, and forced
    # photometry) is re-scored with a kernel matched to that trail.  The
    # velocity *search* stays on the round PSF.  Sensor/optics dependent: the
    # trail is in pixels, so a narrow field (small arcsec/px) trails more per
    # frame for the same angular rate.
    streak_kernel: bool = False
    # Effective per-frame integration time (s) for the streak kernel; the
    # sensor's exposure (continuous readout -> ~1/fps).  0 disables streak
    # modelling even if streak_kernel is true (no exposure -> no trail).
    exposure_time_s: float = 0.0
    # Only build a streak kernel when the trail clears this many pixels;
    # below it the round PSF is used unchanged (negligible elongation, and
    # byte-identical to streak_kernel=false for slow/wide-field cases).
    streak_min_length_px: float = 1.0


@dataclass(frozen=True)
class PipelineConfig:
    """Complete set of pipeline tuning parameters.

    All sections correspond directly to top-level keys in
    opta-model/configs/pipeline_defaults.yaml.  Adding a key to the YAML
    without updating the matching dataclass will be caught by
    tests/test_config.py.
    """

    calibration: CalibrationConfig
    detection: DetectionConfig
    astrometry: AstrometryConfig
    rolling_shutter: RollingShutterConfig
    tracklet: TrackletConfig
    stacking: StackingConfig
    # Optics/hardware profile (opta_model.hardware_catalog profiles.yaml key)
    # the scale/rate-derived gates were resolved against; YAML key
    # ``hardware.profile``.  Carried on the config so downstream consumers can
    # tell which optics a config was built for.
    hardware_profile: str = DEFAULT_HARDWARE_PROFILE

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> PipelineConfig:
        """Load configuration from a YAML file.

        Parameters
        ----------
        path : Path | None
            YAML file path.  Defaults to pipeline_defaults.yaml.
        """
        with open(path or DEFAULTS_PATH, encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        return cls._from_dict(raw)

    @classmethod
    def default(cls) -> PipelineConfig:
        """Return the default configuration from pipeline_defaults.yaml."""
        return cls.from_yaml()

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> PipelineConfig:
        """Coerce YAML section dictionaries into frozen config objects."""
        cal = d["calibration"]
        det = d["detection"]
        ast = d["astrometry"]
        rs = d["rolling_shutter"]
        trk = d["tracklet"]
        stk = d["stacking"]
        hw = d.get("hardware") or {}
        profile = str(hw.get("profile", DEFAULT_HARDWARE_PROFILE))
        # Gates derive from numbers stated in the config; the profile name is
        # only consulted (via opta-model) when a number is missing.
        fps_src: str | float = (
            float(hw["frame_rate_hz"])
            if hw.get("frame_rate_hz") is not None
            else profile
        )
        scale_src: str | float = (
            float(hw["pixel_scale_arcsec"])
            if hw.get("pixel_scale_arcsec") is not None
            else profile
        )
        return cls(
            hardware_profile=profile,
            calibration=CalibrationConfig(
                background_box_size=int(cal["background_box_size"]),
                background_filter_size=int(cal["background_filter_size"]),
            ),
            detection=DetectionConfig(
                snr_threshold=float(det["snr_threshold"]),
                min_streak_pixels=int(det["min_streak_pixels"]),
                max_streak_pixels=int(det["max_streak_pixels"]),
                elongation_threshold=float(det["elongation_threshold"]),
                mask_star_radius_px=int(det["mask_star_radius_px"]),
                mask_veto_max_trajectory_frac=float(
                    det.get("mask_veto_max_trajectory_frac", 0.5)
                ),
            ),
            astrometry=AstrometryConfig(
                max_residual_arcsec=_gate(
                    ast.get("max_residual_arcsec", "derived"),
                    derive_max_residual_arcsec,
                    scale_src,
                ),
                rolling_shutter_correction=bool(ast["rolling_shutter_correction"]),
                sip_distortion=bool(ast.get("sip_distortion", True)),
                skip_flagged_frames=bool(ast.get("skip_flagged_frames", False)),
            ),
            rolling_shutter=RollingShutterConfig(
                enabled=bool(rs["enabled"]),
                row_readout_us=float(rs["row_readout_us"]),
                reference_row=float(rs["reference_row"]),
            ),
            tracklet=TrackletConfig(
                min_detections=int(trk["min_detections"]),
                max_gap_frames=int(trk["max_gap_frames"]),
                linear_fit_residual_arcsec=float(trk["linear_fit_residual_arcsec"]),
                max_sep_arcsec=_gate(
                    trk.get("max_sep_arcsec", "derived"),
                    derive_max_sep_arcsec,
                    fps_src,
                ),
            ),
            stacking=StackingConfig(
                enabled=bool(stk["enabled"]),
                velocity_max_px_s=float(stk["velocity_max_px_s"]),
                velocity_step_px_s=float(stk["velocity_step_px_s"]),
                velocity_min_px_s=float(stk.get("velocity_min_px_s", 0.0)),
                window_duration_s=float(stk.get("window_duration_s", 0.0)),
                window_overlap_s=float(stk.get("window_overlap_s", 0.0)),
                dedup_radius_arcsec=float(stk.get("dedup_radius_arcsec", 30.0)),
                fallback_fps=float(stk.get("fallback_fps", 25.0)),
                psf_sigma_px=float(stk.get("psf_sigma_px", 0.849)),
                far_margin_sigma=float(stk.get("far_margin_sigma", 0.5)),
                coarse_to_fine=bool(stk.get("coarse_to_fine", True)),
                coarse_prethreshold_sigma=float(
                    stk.get("coarse_prethreshold_sigma", 4.0)
                ),
                max_fine_candidates=int(stk.get("max_fine_candidates", 20)),
                seed_window_s=float(stk.get("seed_window_s", 0.0)),
                seed_binning=int(stk.get("seed_binning", 1)),
                refine_snr_min=float(stk.get("refine_snr_min", 7.0)),
                streak_kernel=bool(stk.get("streak_kernel", False)),
                exposure_time_s=float(stk.get("exposure_time_s", 0.0)),
                streak_min_length_px=float(stk.get("streak_min_length_px", 1.0)),
            ),
        )
