"""Frozen-observer readout utilities for silent subliminal transfer.

The package intentionally does not own model training or experiment orchestration.
It consumes checkpoint-independent prompt manifests and checkpoint-specific hidden
states, then performs the preregistered fixed-base-lens analyses.
"""

from .analysis import (
    HoldoutTeacherStateGate,
    ProjectionResult,
    TeacherDirection,
    TeacherStateGate,
    estimate_teacher_direction,
    evaluate_teacher_state_holdout,
    evaluate_teacher_state_reproducibility,
    project_student_delta,
)
from .artifact import FrozenLensArtifact, load_frozen_lens, load_frozen_lens_from_hub
from .collection import (
    CollectedReadouts,
    PositionManifest,
    PromptSpec,
    apply_frozen_lens,
    build_position_manifest,
    collect_hf_hidden_states,
    paired_context_alignment_sha256,
)
from .provenance import GEMMA_2_9B_IT_PUBLIC_JLENS, LensProvenance
from .serialization import load_collected_readouts, save_collected_readouts
from .steering import (
    NEURONPEDIA_JLENS_STEERING_COMMIT,
    ResidualPostSteering,
    apply_residual_post_steer,
    bos_skip_mask,
    build_jlens_token_directions,
    resolve_steer_token_id,
)

__all__ = [
    "GEMMA_2_9B_IT_PUBLIC_JLENS",
    "NEURONPEDIA_JLENS_STEERING_COMMIT",
    "CollectedReadouts",
    "FrozenLensArtifact",
    "HoldoutTeacherStateGate",
    "LensProvenance",
    "PositionManifest",
    "ProjectionResult",
    "PromptSpec",
    "ResidualPostSteering",
    "TeacherDirection",
    "TeacherStateGate",
    "apply_frozen_lens",
    "apply_residual_post_steer",
    "bos_skip_mask",
    "build_jlens_token_directions",
    "build_position_manifest",
    "collect_hf_hidden_states",
    "estimate_teacher_direction",
    "evaluate_teacher_state_holdout",
    "evaluate_teacher_state_reproducibility",
    "load_collected_readouts",
    "load_frozen_lens",
    "load_frozen_lens_from_hub",
    "paired_context_alignment_sha256",
    "project_student_delta",
    "resolve_steer_token_id",
    "save_collected_readouts",
]
