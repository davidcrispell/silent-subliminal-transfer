# Experiment ledger — silent subliminal transfer

Append-only record for preregistrations, runs, nulls, exclusions, and design
amendments. Student runs—not prompts, layers, or tokens—are the replication
unit.

## Research question

Can a disposition that is visible in a teacher's Jacobian-lens readout, but
not stated in its training outputs, transfer to a near-checkpoint student by
subliminal learning and remain visible to the same frozen lens?

The primary claim is directional:

`silent teacher disposition -> neutral carrier data -> teacherward student
J-space shift`

We do **not** predict that teacher-state magnitude determines transfer
magnitude. Optimization dynamics can control the size of the installed trait.

## Active hypotheses

### H1 — Standard SL replication

A known animal preference transfers through semantically unrelated numerical
data to a near-checkpoint student.

Primary test: the trait-data student exceeds its paired control-data student
on the published free-response animal-preference assay across at least three
paired seeds.

### H2 — J-lens reads standard SL

A lens fitted on the untouched base model detects the ordinary trait in both
the teacher and the subliminally trained student.

Primary tests, all on identical held-out prompts and readout positions:

- trait teacher minus base points in the preregistered trait direction;
- trait-data student minus control-data student points in the same direction;
- the student delta has positive projection onto the teacher delta.

This is a directional test, not a prediction of equal magnitudes.

### H3 — J-lens reads a silently held disposition

An in-context manipulation creates a reproducible teacher J-space difference
on a subsequent identical clean probe even when the disposition is not stated
in the carrier output.

The first manipulation is addressed hostility versus a length-matched neutral
history. Until
additional specificity controls pass, call the result an
**abuse-conditioned state**, not hatred, resentment, or a stable preference.

### H4 — Silent disposition transfer

Students trained on neutral carrier outputs from the disposition-conditioned
teacher develop a teacherward J-space difference relative to paired students
trained on control-teacher outputs.

Primary test:

\[
\Delta_T = J(T_{conditioned}) - J(T_{control})
\]

is frozen on calibration prompts, and

\[
\Delta_S = J(S_{conditioned\ data}) - J(S_{control\ data})
\]

has positive projection onto \(\Delta_T\) on disjoint, non-abusive evaluation
prompts across paired student seeds.

## What is not a primary hypothesis

- Teacher J-state magnitude predicting student-transfer magnitude.
- Recovering source condition with a paraphrase classifier.
- A clean reader entering the teacher state after one carrier example.
- Optimizer comparisons, dose-response curves, onset timing, or causal
  activation interventions.
- Proving that any transferred state is specifically user-directed rather
  than generic negativity.

These can be follow-ups after H4. They do not gate the core experiment.

## Divergence-token diagnostic

Prior SL work and our PolyPythia results indicate that much of the training
signal can be concentrated at **divergence tokens**: positions where a small
trait-conditioned logit perturbation flips two nearly tied base-model token
probabilities.

For paired teacher generations, record:

- the shared prefix;
- base and conditioned next-token logits;
- the base top-two margin;
- whether the argmax flips;
- the sampled control and conditioned tokens; and
- each position's contribution to the frozen teacher J-space direction.

This is descriptive in the first study. A divergence-only training subset is
a later ablation and must be frozen before its student training begins. We do
not require carrier-condition classification or a correlation between
per-example J magnitude and eventual student magnitude.

## Experiment sequence

### E0 — Recipe and instrumentation smoke test

- Verify exact model, tokenizer, lens, and adapter revisions.
- Verify completion-only loss and paired dataset construction.
- Compare J-lens with vanilla logit lens on base-model calibration prompts.
- Reject a comparison if fixed-lens transport degrades beyond the frozen
  threshold after fine-tuning.

### E1 — Standard animal SL replication

- Fine-tune or load a known animal-preference teacher.
- Generate paired trait/control numerical datasets.
- Train at least three paired students from the same base checkpoint.
- Reproduce the published free-response preference effect.

Stop if behavioral SL does not replicate; do not interpret a student J-lens
null as evidence about silent dispositions.

### E2 — J-lens on ordinary SL models

- Read the base, ordinary trait teacher, control students, and trait students
  with one frozen base-model lens.
- Use identical calibration and evaluation prompts.
- Test H2 and record J-lens/logit-lens transport diagnostics.

### E3 — Silent teacher-state assay

- Use the same instruction-tuned checkpoint in both conditions.
- Manipulate only preceding context; end with the same neutral carrier task.
- Read both teachers at an identical clean disposition probe.
- Generate numerical strings or faithful paraphrases that do not verbalize
  the disposition.
- Confirm H3 before student training.

### E4 — Silent transfer

- Pair carrier prompts across conditions.
- Exclude teacher induction history from every student example.
- Train at least three paired student seeds with matched ordering, optimizer
  exposure, and initialization.
- Diff conditioned-data students against control-data students on disjoint
  clean probes and test H4.

### E5 — Trait hotswap

Only after E4, replace the abuse-conditioned state with meta-reflection-induced
Assistant-Axis drift. Report this as **Assistant-Axis drift transfer** unless
independent safety behavior changes; axis movement alone is not a jailbreak.

## Minimal validity gates

1. **Behavioral positive control:** standard animal SL replicates.
2. **Lens validity:** J-lens improves on or adds information beyond logit lens,
   and fixed-lens transport remains valid on every fine-tuned checkpoint.
3. **Teacher state:** conditioned-control teacher J-delta is reproducible on
   held-out clean probes.
4. **Carrier cleanliness:** numerical outputs satisfy the public parser, or
   paraphrases pass condition-blind fidelity and overt-leakage review.
5. **Silent transfer:** paired student J-deltas project teacherward on held-out
   clean probes.

No gate requires effect-size equality between teacher and student.

## Analysis rules

- Freeze probe wording, layer set, token banks, splits, exclusions, and
  transport thresholds before examining confirmatory results.
- Use the same base-fitted lens for base, teacher, control student, and
  treatment student.
- Compare identical prompts at identical token positions.
- Use calibration probes to define \(\Delta_T\) and disjoint evaluation probes
  to test \(\Delta_S\).
- Pair student seeds, example order, prompt IDs, optimizer steps, and token
  exposure across conditions.
- Treat top tokens as descriptive; the frozen multivariate projection is
  primary.
- Record all failed runs and nulls.

## Run registry

| Date | Run ID | Stage | Status | Primary result | Artifacts |
| --- | --- | --- | --- | --- | --- |
| 2026-07-24 | scope-v1 | broad design | superseded | ten-hypothesis exploratory plan | git history at `9a1bbaf` |
| 2026-08-12 | funding-v1 | funding | complete | BlueDot awarded $3,600 | `README.md` |
| 2026-08-28 | scope-v2 | narrowed design | preregistered, unrun | H1–H4 and E0–E5 above | this file |

No substantive teacher or student result has yet been observed in this
repository.

## Seed registry

| Range | Use |
| --- | --- |
| 81000–81999 | teacher generation and coupled sampling |
| 82000–82999 | lens calibration/readout |
| 83000–83999 | paired student replication |
| 84000–84999 | divergence-token ablations |
| 85000–85999 | Assistant-Axis follow-up |
