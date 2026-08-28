from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from sst_readout.artifact import load_frozen_lens, sha256_file
from sst_readout.provenance import GEMMA_2_9B_IT_PUBLIC_JLENS, LensProvenance


def synthetic_provenance(**changes) -> LensProvenance:
    value = LensProvenance(
        model_repo="example/model",
        model_revision="1" * 40,
        lens_repo="example/lens",
        lens_revision="2" * 40,
        lens_filename="lens.pt",
        jlens_code_repo="example/code",
        jlens_code_commit="3" * 40,
    )
    return replace(value, **changes)


def test_public_gemma_lens_is_fully_pinned() -> None:
    provenance = GEMMA_2_9B_IT_PUBLIC_JLENS
    provenance.validate()
    assert provenance.expected_size_bytes == 1_053_307_540
    assert provenance.expected_sha256 == (
        "dcc03a24e76205098bd89bc8f2f627c2fe869d516ac31911bb2ab991e5c124a9"
    )
    assert provenance.expected_sidecar_sha256 == (
        "b33fe8346046d54603ab6b021c9949af8cda55843e17c6e2be4c87a7a7722fe8"
    )


def test_load_frozen_lens_validates_and_transports(tmp_path) -> None:
    path = tmp_path / "lens.pt"
    torch.save({"J": {0: 2 * torch.eye(3)}, "n_prompts": 7, "d_model": 3}, path)
    provenance = synthetic_provenance(expected_sha256=sha256_file(path))
    lens = load_frozen_lens(path, provenance, expected_d_model=3, required_layers=[0])
    assert lens.source_layers == (0,)
    assert lens.n_prompts == 7
    assert torch.equal(lens.transport(torch.ones(1, 3), 0), 2 * torch.ones(1, 3))
    assert not lens.jacobian(0).requires_grad


def test_load_frozen_lens_rejects_wrong_digest(tmp_path) -> None:
    path = tmp_path / "lens.pt"
    torch.save({"J": {0: torch.eye(2)}, "n_prompts": 1, "d_model": 2}, path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_frozen_lens(path, synthetic_provenance(expected_sha256="0" * 64))
