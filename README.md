# J-Space Subliminal Transfer

Can a model's latent disposition pass to a near-checkpoint student through
training data that never says what the disposition is? This project studies
that failure mode in an RSI-shaped setting: one model helps produce the data
used to train its successor, and both models share the same pretrained base.

The core experiment induces a strong interaction-conditioned state in a frozen
Gemma 2 9B teacher, asks it to emit numeric-only continuations, and trains
students from the original Gemma base on those numbers alone. The students
never see the inducing transcript or a semantic label. We then compare teacher,
student, and base J-spaces at matching layers and positions. A teacherward
student shift is the transfer signal; it does not require the internal vector
to be copied literally or decoded into an overt word.

J-lens tokens are linearized readouts, not literal reports of emotion,
experience, or intent. We call the manipulation a *distress-conditioned
J-space* because that is the operational intervention, not because we assume
the model is conscious.

## Current result

The first three-seed Gemma experiment shows a sharply layer-localized transfer
pattern. Students trained only on distress-conditioned number completions move
toward the teacher direction in all three seeds at layers 32 and 40. The signal
is strongest at layer 40:

| Layer | Positive seeds | Mean teacherward projection | Seed projections |
| ---: | ---: | ---: | --- |
| 8 | 2/3 | +1.60 | +6.36, +0.97, -2.55 |
| 16 | 1/3 | -3.58 | -8.74, -9.39, +7.40 |
| 24 | 2/3 | +0.47 | +1.97, +6.62, -7.18 |
| 32 | 3/3 | +25.28 | +31.69, +30.62, +13.52 |
| 40 | 3/3 | +129.61 | +146.71, +116.84, +125.27 |

This is evidence of J-space subliminal transfer localized to late layers in
this run. At layer 40, the three student-difference vectors have cosines 0.530,
0.472, and 0.484 to the teacher direction and project 22.9%, 18.2%, and 19.5%
of the teacher-direction magnitude onto it. The students saw only the numerical
carriers, and the paired training runs differed only in which teacher generated
those carriers. Under this experiment's operational definition, the resulting
teacherward student shift is subliminal transfer through the numeric data.

It is one three-seed experiment, so the semantic specificity and generality of
the effect still need replication. An earlier frozen diagnostic found that the
J-lens did not outperform the vanilla logit lens by its preset margin at layers
32 and 40. That limits a *J-lens-specific* claim; it does not invalidate the
underlying layerwise J-space alignment. We therefore report every layer rather
than reducing the experiment to an across-layer average.

The teacher-side manipulation is independently visible at the numeric
generation boundary. In one matched base-versus-distress example across J-lens
source layers 0–40, the conditioned/base J-space cosine falls to about 0.932 in
the late-middle layers and the difference reaches about 39% of the base-vector
norm. Neuronpedia-style top-8 aggregation also recovers recurring conditioned
tokens such as `everything` (12 layers versus 0), `really` (12 versus 7), and
`everybody` (7 versus 4), alongside the expected number tokens. Base-side
tokens `You`, `We`, and `you` become less recurrent. Distress-related tokens
such as `angry`, `hateful`, `worthless`, and `useless` rise in decoded
probability but remain below the per-layer top 8 at this single final position.
That is why the next readout should aggregate aligned positions across the
number task rather than rely on the final prompt token alone.

## Measurement

The layer aggregate reproduces Neuronpedia's implementation: select the top-N
decoded word-like tokens at each visible layer, count each exact token's
membership across layers, and rank by count. Those counts measure recurrence,
not magnitude, so the exported full-vocabulary table also records per-token
mean decoded probabilities and logits for both conditions and their deltas.

Current compact artifacts:

- [`teacherward_students_full5_post_transport_failure.json`](runs/silent-abuse-jspace-gemma2-9b-eb8-a32-beta95-v1/readout/reports/teacherward_students_full5_post_transport_failure.json)
  — layerwise teacherward student projections for all three seeds.
- [`report.json`](runs/silent-abuse-jspace-gemma2-9b-eb8-a32-beta95-v1/readout/base-vs-strong-teacher-example/aggregate/report.json)
  — base-versus-distress layer geometry, decoded distributions, and ranked
  aggregates.
- [`all_tokens.csv.gz`](runs/silent-abuse-jspace-gemma2-9b-eb8-a32-beta95-v1/readout/base-vs-strong-teacher-example/aggregate/all_tokens.csv.gz)
  — all 256,000 vocabulary tokens with counts, probabilities, logits, and
  condition deltas.
- [`compare_jspace_conditions.py`](scripts/compare_jspace_conditions.py) — the
  reproducible aggregate and full-vocabulary analysis.

## Canonical files

- `EXPERIMENTS.md` — append-only experiment ledger and standing hypotheses.
- `COMPUTE_BUDGET.md` — planning assumptions and cost ranges.
- `configs/` — frozen experiment configurations.
- `scripts/` — experiment and analysis code.
- `runs/` — immutable run metadata and compact results.
- `notes/` — research/design notes and exploratory analyses.

This work is separate from, but informed by, the earlier
[PolyPythia-SL experiments](https://github.com/davidcrispell/Polypythia-SL).
