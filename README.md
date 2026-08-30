## Silent Subliminal Transfer

Does the J-space subliminally transfer? This project tests whether an
in-context response tendency that is visible in a teacher's Jacobian-lens
readout can alter neutral number completions and then appear in students
trained on those completions.

The active benign manipulation is direct appreciation and support versus a
word-count-matched neutral history. The earlier hostile-context pilot is
retained for provenance but retired from further carrier generation and
student training. J-lens tokens are measurements of a linearized readout, not
literal reports of emotion, experience, or intent.


It is separate from, but informed by, our previous work on the Tracing the Mechanism of SL project in
`[../replications and experiments 1](https://github.com/davidcrispell/Polypythia-SL/tree/main)`.

## Canonical files

- `EXPERIMENTS.md` — append-only research ledger and standing hypotheses, paper trail
- `COMPUTE_BUDGET.md` — planning assumptions, run counts, and cost ranges, mainly for grant applications
- `configs/` — frozen experiment configurations.
- `scripts/` — executable experiment and analysis code.
- `runs/` — immutable run artifacts and compact reports.
- `notes/` — literature and design notes that are not primary results.

## Current status

- BlueDot funded $3,600 for compute, storage, AI accounts, and contingency.
- The pinned Gemma-2-9B-IT J-lens smoke test passed.
- The prompted-wolf teacher viability gate passed on Gemma-2-9B-IT (61.17%
  wolf exact-first-word rate versus 2.92% unprompted), and the guarded A40
  generation/training benchmark passed.
- The benchmark's measured equal-token pair yield required a documented
  pre-run amendment from 25,000 to 40,000 generated carriers per arm. The
  teacher remains the unchanged checkpoint with a wolf-preference instruction;
  it is not trained.
- After that gate passes, run the warmth-conditioned teacher J-space gate,
  generate paired number carriers, and train paired students.

A large treatment/control teacher readout is only a calibration result. The
smaller treatment-induced change in the carrier-token distribution is the
transmission bottleneck, so teacher readout magnitude is not used to predict
student transfer magnitude.
