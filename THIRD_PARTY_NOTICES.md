# Third-party notices and provenance

This project depends on third-party models, datasets, and software at run time.
Their own licenses and access terms apply; this repository does not redistribute
their weights.

## Jacobian lens

The reference implementation at
[`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens) is
released under the Apache License 2.0. Experiment configurations pin an exact
commit and use it as an external dependency. No copy of that repository is
vendored here.

## Aria H. Wang SL replication

The Gemma recipe is informed by factual settings reported in
[`ariahw/subliminal-learning`](https://github.com/ariahw/subliminal-learning/tree/e4714a4994b597cb87549280e52202a80ebfd2e1)
at commit `e4714a4994b597cb87549280e52202a80ebfd2e1` and its accompanying public
write-up. That commit contains no license file. Accordingly, this repository
does not copy its source code or prompt text; the implementation and prompt
wording here are independent.

## Canonical subliminal-learning recipe

The prompted-teacher design and animal-preference instruction are adapted from
[`MinhxLe/subliminal-learning`](https://github.com/MinhxLe/subliminal-learning/tree/59d4199d30c2d15979e92674044e553c59d6d1fe)
at commit `59d4199d30c2d15979e92674044e553c59d6d1fe`. The configured wolf prompt uses
the grammatical plural “wolves” where the upstream string template's literal
substitution would produce “wolfs.” No source code from that repository is
vendored here.

## Models and artifacts

Google Gemma model weights use the Gemma license and require acceptance of the
upstream access terms. The
[`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens)
artifact repository identifies its license as MIT. The pinned Gemma-2-9B-IT
lens is not redistributed here; its expected LFS SHA-256 is
`dcc03a24e76205098bd89bc8f2f627c2fe869d516ac31911bb2ab991e5c124a9`.
Each run must retain the exact model revision, artifact revision, and verified
checksum in its provenance record.
