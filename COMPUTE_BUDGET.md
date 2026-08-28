# Funded compute plan

Updated 2026-08-28. BlueDot awarded **$3,600** for compute, storage,
AI/API accounts, and contingency.

## Working allocation

| Category | Cap | Purpose |
| --- | ---: | --- |
| GPU compute | $2,400 | Standard SL replication, exact/frozen J-lens work, paired silent-transfer students, reruns. |
| Persistent storage | $400 | Base weights, lens checkpoints, adapters, paired datasets, and compact results. |
| AI/API accounts | $400 | Condition-blind paraphrase review and auxiliary evaluation. |
| Contingency | $400 | Capacity substitutions, tax, failed runs, or one confirmatory rerun. |
| **Total** | **$3,600** | |

Treat these as caps, not spending targets. Record every invoice and GPU-hour in
`runs/COST_LEDGER.csv`.

## Compute-stage envelopes

The $2,400 GPU line is provisionally divided as follows:

| Stage | Envelope | Stop condition |
| --- | ---: | --- |
| E0–E1: 9B recipe/instrument/SL positive control | $250 | Behavioral SL fails after one debug rerun. |
| E2: exact or verified 27B lens work | $850 | One-prompt benchmark extrapolates beyond the remaining cap. |
| E2: ordinary 27B SL and J-lens comparison | $500 | Standard SL fails or fixed-lens transport is invalid. |
| E3–E4: silent teacher and paired students | $600 | Clean teacher J-delta does not replicate. |
| E5: Assistant-Axis hotswap | $200 | Silent-transfer gate does not pass. |
| **GPU total** | **$2,400** | |

Unused money rolls forward; it is not automatically spent on larger models.

## Current Lambda reference prices

Lambda's public on-demand page currently lists:

| Instance | VRAM | Price |
| --- | ---: | ---: |
| GH200 | 96GB | $2.29/hour |
| H100 PCIe | 80GB | $3.29/hour |
| H100 SXM | 80GB | $4.29/hour |
| B200 | 180GB | $6.99/hour |
| A6000 | 48GB | $1.09/hour |

Source: [Lambda on-demand instances](https://lambda.ai/instances). Prices
exclude applicable tax and can change. Lambda bills running instances by the
minute; persistent filesystems continue billing until deleted. See
[Lambda billing](https://docs.lambda.ai/public-cloud/billing/).

GH200 is the default for single-GPU Gemma-2-27B BF16 work if the ARM software
stack passes preflight. H100 PCIe is the x86 fallback. B200 is justified only
when its larger memory or measured throughput lowers total cost.

### Public lens-fit timing anchor

The pinned public Gemma-2-9B-IT lens includes its fit configuration and
convergence log. It was fit on one B200 for 453 prompts at a mean of 13.18
seconds per prompt, or 1.66 recorded GPU-hours total. At Lambda's current B200
list price, that is about $11.60 of uninterrupted instance time. See the
[artifact directory](https://huggingface.co/neuronpedia/jacobian-lens/tree/a4114d7752d11eb546e6cf372213d7e75526d3a1/gemma-2-9b-it/jlens/Salesforce-wikitext).

This is a useful lower-scale anchor, not a 27B estimate: model width, number of
backward passes, memory-driven dimension batch, compilation, and checkpoint
writes all change. The $850 27B line is therefore a stage cap covering fit,
validation, and a failed/restarted attempt—not a forecast that should be spent.

## Mandatory benchmark before each tier

Before approving a full stage, run and record:

1. environment and model-load smoke test;
2. 20 optimizer updates;
3. one generation batch;
4. one complete J-lens fit prompt or one artifact-load/readout pass; and
5. projected stage cost with a 2× uncertainty band.

The exact 27B Jacobian-lens fit is the largest uncertainty because one fit
prompt requires many reverse-mode passes. Never extrapolate its cost from an
ordinary forward pass.

## Storage policy

- Save immutable model revision IDs and download the base once per filesystem.
- Save LoRA adapters rather than merged base-model copies.
- Retain resumable lens checkpoints and provenance sidecars.
- Store projections, selected-token logits, top-k outputs, and recomputable
  metadata—not full-vocabulary logits at every generated position.
- Delete idle instances immediately after artifacts are synced.
- Review filesystem charges weekly and delete the filesystem after final
  archival.

## Scientific spending gates

1. Do not begin silent-transfer student training until standard SL replicates.
2. Do not interpret J-space comparisons unless the fixed base lens passes
   checkpoint transport calibration.
3. Do not train silent students until the teacher manipulation creates a
   clean-probe J-space difference.
4. Do not run the Assistant-Axis hotswap until silent disposition transfer is
   observed or a tightly bounded null makes the comparison independently
   valuable.
5. A LoRA null does not rule out full-parameter SL. Full-parameter 27B training
   requires a separate re-budgeting decision; it is not silently authorized by
   the contingency line.
