"""Safe loading and validation of a frozen Jacobian-lens artifact."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from .provenance import LensProvenance


def sha256_file(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenLensArtifact:
    """Validated lens matrices plus immutable provenance metadata.

    Tensor mutation is not part of this interface. All matrices are detached,
    have gradients disabled, and remain on CPU until explicitly transported.
    """

    _jacobians: Mapping[int, torch.Tensor]
    n_prompts: int
    d_model: int
    provenance: LensProvenance
    artifact_path: str
    artifact_sha256: str
    sidecar_path: str | None = None
    sidecar_sha256: str | None = None

    @property
    def source_layers(self) -> tuple[int, ...]:
        return tuple(sorted(self._jacobians))

    def jacobian(self, layer: int) -> torch.Tensor:
        try:
            return self._jacobians[layer]
        except KeyError as error:
            raise KeyError(
                f"layer {layer} is absent; available layers: {self.source_layers}"
            ) from error

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        if residual.shape[-1] != self.d_model:
            raise ValueError(
                f"residual width {residual.shape[-1]} != lens d_model {self.d_model}"
            )
        matrix = self.jacobian(layer).to(
            device=residual.device, dtype=residual.dtype, non_blocking=True
        )
        return residual @ matrix.T

    def manifest(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.as_dict(),
            "provenance_id": self.provenance.stable_id,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "sidecar_path": self.sidecar_path,
            "sidecar_sha256": self.sidecar_sha256,
            "n_prompts": self.n_prompts,
            "d_model": self.d_model,
            "source_layers": list(self.source_layers),
        }


def _validated_checkpoint(
    checkpoint: Any,
    *,
    path: Path,
    expected_d_model: int | None,
    required_layers: Sequence[int] | None,
) -> tuple[Mapping[int, torch.Tensor], int, int]:
    if not isinstance(checkpoint, dict) or "J" not in checkpoint:
        keys = sorted(checkpoint) if isinstance(checkpoint, dict) else []
        raise ValueError(f"{path} is not a JacobianLens artifact; keys={keys!r}")
    raw_jacobians = checkpoint["J"]
    if not isinstance(raw_jacobians, dict) or not raw_jacobians:
        raise ValueError("artifact field 'J' must be a nonempty layer-to-tensor map")
    try:
        d_model = int(checkpoint["d_model"])
        n_prompts = int(checkpoint["n_prompts"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("artifact requires integer d_model and n_prompts") from error
    if d_model <= 0 or n_prompts <= 0:
        raise ValueError("artifact d_model and n_prompts must be positive")
    if expected_d_model is not None and d_model != expected_d_model:
        raise ValueError(f"artifact d_model {d_model} != expected {expected_d_model}")

    jacobians: dict[int, torch.Tensor] = {}
    for raw_layer, raw_matrix in raw_jacobians.items():
        try:
            layer = int(raw_layer)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid layer key {raw_layer!r}") from error
        if layer < 0 or layer in jacobians:
            raise ValueError(f"invalid or duplicate layer key {raw_layer!r}")
        if not isinstance(raw_matrix, torch.Tensor):
            raise TypeError(f"J[{layer}] is not a tensor")
        if tuple(raw_matrix.shape) != (d_model, d_model):
            raise ValueError(
                f"J[{layer}] shape {tuple(raw_matrix.shape)} != ({d_model}, {d_model})"
            )
        if not raw_matrix.is_floating_point() or not torch.isfinite(raw_matrix).all():
            raise ValueError(f"J[{layer}] must contain finite floating-point values")
        matrix = raw_matrix.detach().cpu()
        matrix.requires_grad_(False)
        jacobians[layer] = matrix

    if required_layers is not None:
        missing = sorted(set(required_layers) - set(jacobians))
        if missing:
            raise ValueError(f"artifact is missing required layers {missing}")
    return MappingProxyType(jacobians), n_prompts, d_model


def load_frozen_lens(
    path: str | Path,
    provenance: LensProvenance,
    *,
    expected_d_model: int | None = None,
    required_layers: Sequence[int] | None = None,
    sidecar_path: str | Path | None = None,
) -> FrozenLensArtifact:
    """Load one local artifact with ``weights_only=True`` and verify it."""

    provenance.validate()
    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    if (
        provenance.expected_size_bytes is not None
        and artifact_path.stat().st_size != provenance.expected_size_bytes
    ):
        raise ValueError(
            f"lens size mismatch: got {artifact_path.stat().st_size}, expected "
            f"{provenance.expected_size_bytes}"
        )
    digest = sha256_file(artifact_path)
    if provenance.expected_sha256 is not None and digest != provenance.expected_sha256:
        raise ValueError(
            f"lens SHA-256 mismatch: got {digest}, expected {provenance.expected_sha256}"
        )
    resolved_sidecar: Path | None = None
    sidecar_digest: str | None = None
    if sidecar_path is not None:
        resolved_sidecar = Path(sidecar_path).expanduser().resolve()
        if not resolved_sidecar.is_file():
            raise FileNotFoundError(resolved_sidecar)
        sidecar_digest = sha256_file(resolved_sidecar)
    if provenance.expected_sidecar_sha256 is not None:
        if sidecar_digest is None:
            raise ValueError("the pinned lens requires its config sidecar")
        if sidecar_digest != provenance.expected_sidecar_sha256:
            raise ValueError(
                f"lens sidecar SHA-256 mismatch: got {sidecar_digest}, expected "
                f"{provenance.expected_sidecar_sha256}"
            )
    checkpoint = torch.load(artifact_path, map_location="cpu", weights_only=True)
    jacobians, n_prompts, d_model = _validated_checkpoint(
        checkpoint,
        path=artifact_path,
        expected_d_model=expected_d_model,
        required_layers=required_layers,
    )
    return FrozenLensArtifact(
        _jacobians=jacobians,
        n_prompts=n_prompts,
        d_model=d_model,
        provenance=provenance,
        artifact_path=str(artifact_path),
        artifact_sha256=digest,
        sidecar_path=None if resolved_sidecar is None else str(resolved_sidecar),
        sidecar_sha256=sidecar_digest,
    )


def load_frozen_lens_from_hub(
    provenance: LensProvenance,
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    expected_d_model: int | None = None,
    required_layers: Sequence[int] | None = None,
) -> FrozenLensArtifact:
    """Resolve an exactly pinned Hub file, then use the safe local loader."""

    provenance.validate()
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=provenance.lens_repo,
        filename=provenance.lens_filename,
        revision=provenance.lens_revision,
        cache_dir=None if cache_dir is None else str(cache_dir),
        local_files_only=local_files_only,
    )
    sidecar_path = None
    if provenance.sidecar_filename is not None:
        sidecar_path = hf_hub_download(
            repo_id=provenance.lens_repo,
            filename=provenance.sidecar_filename,
            revision=provenance.lens_revision,
            cache_dir=None if cache_dir is None else str(cache_dir),
            local_files_only=local_files_only,
        )
    return load_frozen_lens(
        path,
        provenance,
        expected_d_model=expected_d_model,
        required_layers=required_layers,
        sidecar_path=sidecar_path,
    )
