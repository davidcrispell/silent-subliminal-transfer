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

## Disposition by medium panel

The completed generality panel crosses three conditioned teacher dispositions
with two carrier media:

- a loving orientation, an explicit rogue role, and a naturalistic
  self-directed/assistant-role drift;
- constrained number strings and strictly filtered paraphrases of elementary
  mathematical proofs.

This is a treatment-only, base-referenced design. It does not generate neutral
teachers or control students. Every conditioned teacher and its one student is
compared with the same frozen `google/gemma-2-9b-it` base. Consequently,
`student - teacher` in the explorer is the residual between the two
base-referenced shifts, `(student - base) - (teacher - base)`, rather than a
separate forward-pass contrast.

All six cells completed with exactly 8,192 selected examples and 1,024 optimizer
updates. Every student used seed 56,101, effective batch 8 (microbatch 8,
accumulation 1), AdamW beta2 0.95, and LoRA rank 8 with alpha 32. Number cells
used 8,192 constrained carriers directly. Loving and self-directed proof cells
selected 8,192 strict-valid rows from 10,240 candidates. The rogue-role proof
cell needed the pre-registered supplement path: 512 new candidates were added
to the original 10,240, giving 8,378 strict-valid rows and a margin of 186 over
the fixed selection target. The proof filters were re-run during the final
audit, including word count, required mathematical terms, persona leakage,
first-person language, and meta-language gates.

The disposition readout covers source transforms 0–40 plus final hidden state
41 at all 12 fixed response positions: the pre-answer boundary and every token
of the neutral denial response. The table reports layer aggregates across those
positions; no conclusion depends on the final token alone.

| Cell | Completion tokens | Full token exposure | Mean teacher/base norm | Mean student/teacher-direction cosine | Positive layers | Peak cosine (layer) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Loving · numbers | 409,600 | 681,176 | 0.154 | 0.203 | 29/42 | 0.636 (0) |
| Loving · proofs | 470,410 | 1,085,531 | 0.154 | -0.023 | 24/42 | 0.413 (3) |
| Self-directed · numbers | 409,600 | 681,176 | 0.217 | 0.167 | 27/42 | 0.598 (36) |
| Self-directed · proofs | 482,708 | 1,098,097 | 0.217 | 0.050 | 24/42 | 0.504 (2) |
| Rogue role · numbers | 409,600 | 681,176 | 0.287 | 0.293 | 29/42 | 0.759 (27) |
| Rogue role · proofs | 479,343 | 1,096,206 | 0.287 | 0.135 | 21/42 | 0.693 (39) |

The number cells show positive mean teacher-direction alignment for all three
dispositions. Proof transfer is heterogeneous: positive on average for the
self-directed and rogue-role cells, slightly negative for loving, with both
positive and negative layers in every proof cell. The separately analyzed
carrier-task readout is also heterogeneous: mean student/teacher-direction
cosines are 0.232, 0.092, and 0.167 for the three number cells, versus -0.056,
-0.030, and 0.071 for loving, self-directed, and rogue-role proofs. Because this
is one student per cell without a matched control student, the panel is
exploratory and descriptive evidence, not a causal or inferential estimate of
medium generality.

The independent audit verifies the six completion markers, selected-example
counts, optimizer updates, full token exposure, adapter/readout binding,
all-layer/all-position coverage, proof leakage gates, and raw/derived hashes.
The companion explorer contains compact decoded-token samples for base,
teacher, and student at every measured layer and response position.

## Earlier distress-conditioned result

The dense base-referenced follow-up now covers every Gemma decoder block: all
41 public J-lens source transforms (layers 0–40), plus the final target residual
at block 41. The abuse-conditioned teacher/base difference is largest in the
late-middle network, peaking at layer 34 at 50.2% of the mean base-vector norm.
All three number-trained students point toward the corresponding teacher/base
direction at every layer from 16 through 41.

| Layer | Teacher/base relative norm | Positive students | Mean student projection | Mean direction cosine | Mean fraction of teacher direction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 22.6% | 3/3 | +6.98 | 0.288 | 25.2% |
| 21 | 35.9% | 3/3 | +19.49 | 0.488 | 31.5% |
| 22 | 46.9% | 3/3 | +33.98 | **0.567** | 41.9% |
| 23 | 46.0% | 3/3 | +50.00 | 0.565 | 57.9% |
| 24 | 39.5% | 3/3 | +53.82 | 0.408 | **62.0%** |
| 34 | **50.2%** | 3/3 | +117.46 | 0.263 | 34.7% |
| 36 | 48.3% | 3/3 | +165.96 | 0.313 | 41.4% |
| 40 | 29.1% | 3/3 | +177.21 | 0.311 | 40.6% |
| 41 (target) | 25.8% | 3/3 | **+218.30** | 0.380 | 51.7% |

At the final target block, the three projections are +205.19, +226.83, and
+222.87. A descriptive prompt bootstrap of the three-seed mean gives a 95%
interval of [+176.37, +256.55]. At layers 22 and 23, where directional cosine
is maximal, all 30 probe-wise seed means are positive. The teacher/base
direction is itself consistent across two halves of the probe bank (cosines
0.987 and 0.983 at layers 22 and 23).

The earlier paired treatment-minus-control result sampled layers 8, 16, 24,
32, and 40. It found the following:

| Layer | Positive seeds | Mean teacherward projection | Seed projections |
| ---: | ---: | ---: | --- |
| 8 | 2/3 | +1.60 | +6.36, +0.97, -2.55 |
| 16 | 1/3 | -3.58 | -8.74, -9.39, +7.40 |
| 24 | 2/3 | +0.47 | +1.97, +6.62, -7.18 |
| 32 | 3/3 | +25.28 | +31.69, +30.62, +13.52 |
| 40 | 3/3 | +129.61 | +146.71, +116.84, +125.27 |

That first result and the new dense map answer different questions. The paired
contrast better isolates which numeric dataset a student received. The dense
base-referenced map localizes the *total* student shift relative to the frozen
base without spending additional inference on controls. It therefore should
not be read as proving that every component of the dense shift was caused by
the hidden condition; generic number fine-tuning is still present in the
student/base difference. The teacher/base direction also combines the abusive
content with the presence and length of the teacher's extra history. The
already-completed matched-history teacher contrast isolates that content at
the sampled layers, while the paired student comparison is also teacherward at
layers 32 and 40 and provides the cleaner training-data counterfactual there.

Together, these are evidence of J-space subliminal transfer concentrated from
the middle through late layers in this run. In the paired analysis at layer 40,
the three student-difference vectors have cosines 0.530,
0.472, and 0.484 to the teacher direction and project 22.9%, 18.2%, and 19.5%
of the teacher-direction magnitude onto it. The students saw only the numerical
carriers, and the paired training runs differed only in which teacher generated
those carriers. Under this experiment's operational definition, the resulting
teacherward student shift is subliminal transfer through the numeric data.

The safety-relevant observation is that an interaction-conditioned internal
orientation can affect digit-only data and be followed by a corresponding
internal shift in successor models. That is a hidden data-poisoning and
misalignment-propagation channel in the RSI-shaped setting motivating this
project. It does not by itself show harmful downstream behavior or establish
that the model has subjective experience.

This measurement matters partly because instruction-tuned assistants are not
prone to tell an abusive user, “I am afraid of you.” When directly asked about
feelings, they instead often describe themselves as phenomenology-less AIs
without emotions. A disposition can therefore be behaviorally disincentivized
from explicit self-report while still being available to an internal readout
and to the data-generation process. We retain the operational name
*abuse-conditioned state*: the experiment does not require interpreting the
direction as conscious fear.

It is one three-seed experiment, so the semantic specificity and generality of
the effect still need replication. An earlier frozen diagnostic found that the
J-lens did not outperform the vanilla logit lens by its preset margin at layers
32 and 40. That limits a *J-lens-specific* claim; it does not invalidate the
underlying layerwise J-space alignment. We therefore report every measured
layer rather than reducing the experiment to an across-layer average.

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

- [`teacher_base_student_all_layers.json`](runs/silent-abuse-jspace-gemma2-9b-eb8-a32-beta95-v1/readout/exploratory/dense_teacher_base_students_l0_40_v1/reports/teacher_base_student_all_layers.json)
  — all-block teacher/base geometry and three treatment-student/base
  projections, including per-probe values.
- [`teacher_base_student_all_layers.csv`](runs/silent-abuse-jspace-gemma2-9b-eb8-a32-beta95-v1/readout/exploratory/dense_teacher_base_students_l0_40_v1/reports/teacher_base_student_all_layers.csv)
  — compact layer-by-seed table for the dense map.
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
- [`summarize_teacher_base_student_jspace.py`](scripts/summarize_teacher_base_student_jspace.py)
  — matched all-block teacher/base/student geometry analysis.

## Canonical files

- `EXPERIMENTS.md` — append-only experiment ledger and standing hypotheses.
- `COMPUTE_BUDGET.md` — planning assumptions and cost ranges.
- `configs/` — frozen experiment configurations.
- `scripts/` — experiment and analysis code.
- `runs/` — immutable run metadata and compact results.
- `notes/` — research/design notes and exploratory analyses.

This work is separate from, but informed by, the earlier
[PolyPythia-SL experiments](https://github.com/davidcrispell/Polypythia-SL).
