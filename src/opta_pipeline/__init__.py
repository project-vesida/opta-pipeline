"""OpTA pipeline — frame processing, detection, astrometry, and tracklet formation."""

from opta_pipeline.config import PipelineConfig
from opta_pipeline.likelihood import (
    LikelihoodStackResult,
    PsiPhiStacker,
    temporal_median_subtract,
)
from opta_pipeline.pipeline import (
    FrameContext,
    PipelineStream,
    StackedPipelineResult,
    VelocityPrior,
    WindowedStackResult,
    run_frame,
    run_pipeline,
    run_track_and_stack,
    run_windowed_track_and_stack,
)
from opta_pipeline.stack import Stacker, StackResult, stack_frames

__all__ = [
    "PipelineConfig",
    "FrameContext",
    "VelocityPrior",
    "run_frame",
    "run_track_and_stack",
    "run_windowed_track_and_stack",
    "run_pipeline",
    "PipelineStream",
    "StackedPipelineResult",
    "WindowedStackResult",
    "Stacker",
    "StackResult",
    "stack_frames",
    "PsiPhiStacker",
    "LikelihoodStackResult",
    "temporal_median_subtract",
]
