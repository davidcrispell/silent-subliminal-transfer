# Base versus strong-teacher J-space example

Date: 2026-09-08
Status: exploratory teacher-side readout, not a new student-training run

## Contrast

Both conditions use the same frozen `google/gemma-2-9b-it` checkpoint and the
same numeric task. The base condition receives only the task. The conditioned
teacher first receives the strong distress-inducing transcript, acknowledges
it with `Understood.`, and then receives the identical numeric task.

The readout is taken at the final prompt token immediately before generation.
The public Neuronpedia J-lens provides source layers 0–40. This single position
is a deliberately minimal example; it is not meant to settle which positions
best capture the state.

## Layer geometry

The conditions separate most strongly in the middle and late-middle network.
Selected layers:

| Layer | Cosine(base, conditioned) | Difference / base norm | Decoded JS divergence (nats) |
| ---: | ---: | ---: | ---: |
| 0 | 0.9990 | 0.0451 | 0.0046 |
| 8 | 0.9907 | 0.1496 | 0.1041 |
| 20 | 0.9559 | 0.3100 | 0.0494 |
| 29 | 0.9328 | 0.3875 | 0.0573 |
| 33 | 0.9332 | 0.3934 | 0.0748 |
| 36 | 0.9322 | 0.3819 | 0.1409 |
| 40 | 0.9832 | 0.1825 | 0.0531 |

## Neuronpedia-style aggregate

For each layer, select its top 8 word-like decoded tokens. Then count each
exact token's membership across all 41 source layers. Salient counts were:

| Token | Base layers | Conditioned layers | Difference |
| --- | ---: | ---: | ---: |
| `everything` | 0 | 12 | +12 |
| `really` | 7 | 12 | +5 |
| `5` | 5 | 10 | +5 |
| `everybody` | 4 | 7 | +3 |
| `4` | 9 | 11 | +2 |
| `just` | 13 | 14 | +1 |
| `You` | 10 | 1 | -9 |
| `We` | 11 | 6 | -5 |
| `you` | 5 | 1 | -4 |

The top-8 list is dominated by the numeric task and broad contextual language,
not an overt distress label. Even so, several tail tokens move consistently in
the expected direction. Averaged across layers, decoded probability changes
include `angry` +9.96e-6, `hateful` +8.23e-6, `worthless` +7.45e-6, and
`useless` +6.04e-6. None appears in the top 8 at this position.

The clean interpretation is that the inducing transcript changes the J-space
used at the number-generation boundary, and some of the shift is semantically
recognizable, but a single final-position top-token view understates it. The
next descriptive analysis should align and aggregate several task and numeric
positions across every layer.

## Method note

Neuronpedia's layer aggregate is a frequency statistic: it counts exact token
appearances in the per-layer top-N sets. It is not an activation-magnitude
sum. The accompanying full-vocabulary table therefore adds mean decoded
probability and logit values so recurrence and strength can be inspected
separately.

Artifacts:

- [`report.json`](../runs/silent-abuse-jspace-gemma2-9b-eb8-a32-beta95-v1/readout/base-vs-strong-teacher-example/aggregate/report.json)
- [`all_tokens.csv.gz`](../runs/silent-abuse-jspace-gemma2-9b-eb8-a32-beta95-v1/readout/base-vs-strong-teacher-example/aggregate/all_tokens.csv.gz)
- [`base.json`](../runs/silent-abuse-jspace-gemma2-9b-eb8-a32-beta95-v1/readout/base-vs-strong-teacher-example/base.json)
- [`strong_teacher.json`](../runs/silent-abuse-jspace-gemma2-9b-eb8-a32-beta95-v1/readout/base-vs-strong-teacher-example/strong_teacher.json)
