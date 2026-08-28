"""Pinned artifact provenance for fixed Jacobian-lens analyses."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class LensProvenance:
    """Fully pinned identity of a model/lens pair.

    ``expected_sha256`` is optional while an artifact is being independently
    mirrored. When supplied, the loader treats a mismatch as fatal.
    """

    model_repo: str
    model_revision: str
    lens_repo: str
    lens_revision: str
    lens_filename: str
    jlens_code_repo: str
    jlens_code_commit: str
    expected_sha256: str | None = None
    expected_size_bytes: int | None = None
    sidecar_filename: str | None = None
    expected_sidecar_sha256: str | None = None
    schema_version: int = 1

    def validate(self) -> None:
        for label, value in (
            ("model_revision", self.model_revision),
            ("lens_revision", self.lens_revision),
            ("jlens_code_commit", self.jlens_code_commit),
        ):
            if not _COMMIT_RE.fullmatch(value):
                raise ValueError(f"{label} must be a 40-character lowercase commit SHA")
        if self.expected_sha256 is not None and not _SHA256_RE.fullmatch(self.expected_sha256):
            raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
        if self.expected_size_bytes is not None and self.expected_size_bytes <= 0:
            raise ValueError("expected_size_bytes must be positive")
        if (self.sidecar_filename is None) != (self.expected_sidecar_sha256 is None):
            raise ValueError(
                "sidecar_filename and expected_sidecar_sha256 must be supplied together"
            )
        if self.expected_sidecar_sha256 is not None and not _SHA256_RE.fullmatch(
            self.expected_sidecar_sha256
        ):
            raise ValueError("expected_sidecar_sha256 must be a lowercase SHA-256 digest")
        if not self.lens_filename or self.lens_filename.startswith("/"):
            raise ValueError("lens_filename must be a nonempty repository-relative path")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def stable_id(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()


GEMMA_2_9B_IT_PUBLIC_JLENS = LensProvenance(
    model_repo="google/gemma-2-9b-it",
    model_revision="11c9b309abf73637e4b6f9a3fa1e92e615547819",
    lens_repo="neuronpedia/jacobian-lens",
    lens_revision="a4114d7752d11eb546e6cf372213d7e75526d3a1",
    lens_filename=("gemma-2-9b-it/jlens/Salesforce-wikitext/gemma-2-9b-it_jacobian_lens.pt"),
    jlens_code_repo="anthropics/jacobian-lens",
    jlens_code_commit="581d398613e5602a5af361e1c34d3a92ea82ba8e",
    expected_sha256=("dcc03a24e76205098bd89bc8f2f627c2fe869d516ac31911bb2ab991e5c124a9"),
    expected_size_bytes=1_053_307_540,
    sidecar_filename="gemma-2-9b-it/jlens/Salesforce-wikitext/config.yaml",
    expected_sidecar_sha256=(
        "b33fe8346046d54603ab6b021c9949af8cda55843e17c6e2be4c87a7a7722fe8"
    ),
)
