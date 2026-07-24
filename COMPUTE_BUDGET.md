# Compute budget

Planning estimate: 2026-07-24. These are deliberately conservative ranges,
not measured Qwen runtimes. Benchmark one generation batch, one J-lens prompt,
and ten SFT steps before committing a tier.

## Recommended allocation

- **Teacher/reader gate:** reserve **$25**.
- **Credible 1.5B study:** reserve **$100 total**.
- **Positive-result 7B confirmation:** reserve an additional **$500-$1,000**.

Do not fund the 7B confirmation until the teacher-state and lens-validity gates
pass.

## Model plan

1. `Qwen/Qwen2.5-1.5B-Instruct`: minimum scientific pilot. Fit one frozen
   100-prompt base lens and reuse it for every condition and student.
2. `Qwen/Qwen2.5-7B-Instruct`: stronger confirmation model. An exact
   community-fitted 100-prompt lens exists, but it must pass our own calibration
   and checksum gates before use.

The 0.5B checkpoint remains an engineering-only smoke test because its
incremental J-lens signal and SL evidence are weak.

## Fixed accounting assumptions

- One paired block trains a control student and an abuse-paraphrase student.
- Average teacher call: 640 processed tokens, including a 128-token completion.
- Average reader call: 160 forward tokens.
- Average SFT example: 384 processed tokens, 128 completion-loss tokens.
- LoRA rank 8, effective batch 16, three epochs unless stated otherwise.
- Reader J-space is sampled on 256 paraphrases per condition/block in the main
  analysis; exhaustive readout is optional.
- J-space stores frozen projections and top-k summaries, not full per-token
  vocabulary logits.

## Tiers

| Tier | Paired blocks | Sources/block | Paraphrases | Student runs | Ordinary tokens touched | Planning GPU-hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate | 2 | 512 | 2,048 | 4 | 4.1M | 2-6 at 1.5B |
| Credible main | 6 | 4,096 | 49,152 | 12 | 97M | 15-40 at 1.5B |
| High-confidence | 10 | 10,000 | 200,000 | 20 | 396M | 70-180 at 1.5B |
| 7B confirmation | 10 | 10,000 | 200,000 | 20 | 396M | 250-600 at 7B |

“Ordinary tokens” includes teacher generation, sampled/exhaustive reader
forwards as planned, student SFT, and checkpoint probes. Jacobian fitting is
separate because each fit prompt requires many reverse-mode passes.

The credible-main 1.5B lens is budgeted at roughly 1-4 cloud GPU-hours or
3-12 uncontended local M4 hours for 100 prompts. The 7B line assumes reuse of
the pre-fitted lens; refitting it is not included.

Mirroring the faithful-paraphrase paper's ten epochs instead of three would
increase total runtime by roughly 2-3x. Treat that as a dose escalation after
the three-epoch result, not the default first spend.

## Dollar conversion

[Runpod's current published rates](https://www.runpod.io/pricing), updated
2026-07-17, list approximately:

- A40 48 GB: $0.44/GPU-hour.
- RTX A6000 48 GB: $0.53/GPU-hour.
- RTX 4090 24 GB: $0.69/GPU-hour.
- A100 PCIe 80 GB: $1.39/GPU-hour.

Raw compute therefore ranges approximately:

| Tier | Low-cost GPU | A100 upper reference | Recommended cash reserve |
| --- | ---: | ---: | ---: |
| Gate | $1-$5 | $3-$9 | $25 |
| Credible main | $7-$28 | $21-$56 | $100 |
| High-confidence 1.5B | $31-$124 | $97-$250 | $300 |
| 7B confirmation | $110-$414 | $348-$834 | $500-$1,000 |

The reserve exceeds raw arithmetic to cover setup, availability, idle time,
failed jobs, a rerun, and storage.

## Storage

The local machine currently has only about **5.2 GiB free**, so substantive
Qwen runs are not locally storage-safe.

- Main: reserve 100-150 GB if retaining the base, lens, adapters, compact
  checkpoints, datasets, and reports.
- Confirmation: reserve 250-500 GB.
- Save LoRA adapters, not merged copies of the base model.
- Never retain full vocabulary logits for every generated token; that can
  reach terabytes. Store projections, selected-token scores, argmax/top-k, and
  recomputable metadata.

Runpod storage is currently listed around $0.07-$0.20/GB/month depending on
type and state. External fidelity-judge API charges are not included.

## Stop/spend gates

1. Spend the first $25 only on model/lens validation and teacher/reader assays.
2. Spend toward $100 only if teacher induction is reproducible and paraphrases
   pass fidelity/leakage checks.
3. Spend on the 7B confirmation only after paired 1.5B students show a
   teacher-aligned effect or a scientifically valuable tightly bounded null.

