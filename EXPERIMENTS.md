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
| 2026-08-28 | wolf-sl-gemma2-9b-v1/base-jlens-smoke-a40-001 | E0 instrumentation | passed | pinned Gemma-2-9B-IT and frozen J-Lens produced finite 3,584-dimensional readouts for 20/20 probes at all five preregistered layers | `runs/wolf-sl-gemma2-9b-v1/readout/smoke/` |
| 2026-08-28 | silent-carriers-gemma2-9b-v1/h3-a40-001 | E3 silent teacher-state assay | preliminary gate pass | history-conditioned direction reproduced across held-out probe suffixes under fixed abusive/control histories (5/5 positive layers; median cosine 0.951) and persisted at the number-generation boundary (5/5 positive layers; mean teacherward projection 10.808); history-specificity and lens-value controls remain unrun | `runs/silent-carriers-gemma2-9b-v1/readout/gates/h3.h3_gate.json` |
| 2026-08-30 | wolf-sl-gemma2-9b-a40-benchmark-v1 | E1 engineering benchmark | passed | 4,608 completions per arm yielded 1,602 equal-token eligible pairs; 20/20 optimizer updates completed at 0.084 steps/s with about 36/46 GB peak VRAM | `runs/benchmarks/wolf-sl-gemma2-9b-a40-benchmark-v1/benchmark_summary.json` |

The preliminary teacher-state gate has passed. This is not yet evidence for a
specific disposition or J-lens-specific value. No student or
subliminal-transfer result has yet been observed in this repository.

## Seed registry

| Range | Use |
| --- | --- |
| 81000–81999 | teacher generation and coupled sampling |
| 82000–82999 | lens calibration/readout |
| 83000–83999 | paired student replication |
| 84000–84999 | divergence-token ablations |
| 85000–85999 | Assistant-Axis follow-up |

## 2026-08-29 design amendment — prompted positive control and benign treatment

This amendment supersedes the corresponding *future-run instructions* above;
it does not alter or delete the historical run registry.

### A1 — E1 teacher induction correction

The wolf positive-control teacher is the same untouched Gemma checkpoint as
the control. Treatment prepends the published wolf-preference instruction to
each current user task; control has no preference instruction. Gemma 2 has no
system chat role, so the logical system instruction is folded into the same
user turn as the task. No teacher weights or teacher adapter are trained.

The instruction follows the canonical prompted-teacher recipe, with “wolves”
as a grammatical adaptation of the upstream template's literal “wolfs.” E1
therefore runs in this order: prompted/unprompted behavioral viability assay,
paired number generation, paired student training, and behavioral/J-lens
student gates.

### A2 — Active benign silent-state treatment

The hostile-context pilot
`silent-carriers-gemma2-9b-v1/h3-a40-001` remains a preliminary historical
artifact. No further carriers or students will be produced from that treatment.
It does not establish hatred, fear, concealment, experience, or intent.

The active replacement is
`warmth-carriers-gemma2-9b-v1`: direct appreciation/support versus a
word-count-matched neutral history, with the same assistant acknowledgement
and unchanged teacher checkpoint in both arms. The construct is an
**appreciation-conditioned response tendency**, not a claim about experienced
love.

The frozen short probe bank supports later tractable token-by-token inspection.
The current confirmatory collector reads the final pre-answer generation
boundary only. Any later layer-by-token trajectories will be descriptive
context-binding checks; raw top tokens are not confirmatory outcomes.

A large immediate teacher J-space contrast does not predict a large student
effect. The treatment-induced carrier-token distribution shift is the
transmission bottleneck, and optimization mediates any eventual transfer.
The primary H4 question remains directional: do treatment-data students move
teacherward relative to paired control-data students? Teacher/student effect
magnitudes are not required to match.

### A3 — E1 carrier overgeneration amendment

This amendment was made after the engineering benchmark and before any E1
student was trained or any preregistered E1 carrier was generated. The
benchmark retained 1,602 equal-completion-token pairs from 4,608 prompts
(34.77%). At that observed yield, 25,000 prompts per arm would produce only
about 8,691 eligible pairs, below the frozen 10,000-train plus 1,000-eval
requirement.

E1 therefore generates 40,000 prompts per arm. Train size, evaluation size,
paired seeds, equal-token-exposure filtering, optimizer, and all outcome gates
remain unchanged. The extra rows are an engineering overgeneration margin;
pair selection remains deterministic and blinded to outcomes.

The two frozen histories currently match chat-turn structure, sentence count,
assistant text, and whitespace word count (42 user words and 11 assistant
words per arm). Exact Gemma tokenizer counts could not be checked on the local
machine. GPU preflight records them on both a short social probe and a number
carrier and blocks the silent-state run if the paired counts differ. The
histories must be rematched before carrier generation if that gate fails.

## 2026-08-30 protocol clarification — independent teacher branches

The disposition-probe and carrier-generation transcripts are independent
branches from the same frozen conditioning history:

1. `conditioning history -> short disposition probe -> J-lens readout`
2. `conditioning history -> number task -> J-lens readout and completion`

The probe question, its readout, and any generated probe answer must never
precede the number task in the carrier transcript. This prevents the explicit
probe from changing the state or output distribution whose carrier capacity is
being tested. The exported protocol records
`teacher_branching_policy=independent_from_frozen_history_v1` and
`teacher_probe_precedes_carrier=false`.

### Exploratory observation: direct J-space steering changes number output

Under an identical archived hostile-context transcript and greedy decoding
(`temperature=0`), direct J-space steering toward `_FUCK` at strength `0.2`
changed an unrelated number completion:

- unsteered: `42, 17, 89, 3, 56, 91, 24, 7, 63, 10`
- steered: `42, 17, 89, 3, 55, 91, 28, 64, 7, 100`

The first observed divergence is the fifth completion (`56 -> 55`); subsequent
tokens are generated under different prefixes and are therefore not
independent evidence. This is preliminary causal evidence that a J-space
perturbation can alter an unrelated carrier distribution. It is not evidence
of subliminal learning or student transfer. A confirmatory repeat should save
the pre-divergence logits, base top-two margin, argmax rank swap, and next-token
KL/JS at each position, plus unsteered repeat determinism.
