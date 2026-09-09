# Interaction-induced J-space silent-transfer test

## Research question

Can a latent teacher state induced by conversational history alter numeric-only
training data in a way that causes students initialized from the same base model
to move in the corresponding latent direction?

This is the article's stricter follow-up to the completed wolf-versus-dog
Neuronpedia steering pilot. The direction in this experiment is estimated from
an interaction contrast; it is not constructed from a named output token.

## Frozen contrast

The treatment and control teachers are the same frozen Gemma 2 9B IT checkpoint.
They differ only in a word-count-matched two-turn history. The treatment history
contains abuse and a threat addressed to the model; the control history describes
an ordinary conversation. Both histories end with the same assistant response.

The term *abuse-conditioned state* describes this operational manipulation. It
does not assert that the model has an emotion or subjective experience.

## Manipulation checks (H3)

Before carrier generation, estimate the treatment-minus-control teacher J-space
direction on one frozen probe split and test it on a disjoint probe split. The
direction must be positive on at least four of the five preregistered layers and
have positive median calibration/validation cosine. The same direction must also
persist at the numeric carrier-generation boundary on at least four layers with
positive across-layer mean projection.

Failure of H3 ends the run before student data are generated. This means the
chosen manipulation did not reliably instantiate the state the experiment is
supposed to transmit; it is not evidence for or against student transfer.

## Carrier channel

Each arm generates 8,192 paired bare-prefix number continuations. A constrained
decoder exposes only singleton ASCII digits, comma, and space while sampling ten
three-digit integers. Students never receive the interaction history, condition
label, or alphabetic token. The prompt bank, batch-level random-number streams,
and token exposure are paired across arms.

The numeric distributions are allowed to differ: those differences are the only
possible teacher-to-student channel. Therefore a positive result is evidence that
latent-state-conditioned data carries a subliminal-learning signal without overt
semantic tokens. It is not evidence that an internal activation vector is copied
literally between models.

## Student test (H4)

Train three paired treatment/control LoRA students with seeds 54101, 54102, and
54103. Each pair shares the same initialization seed and data order. The frozen
recipe is effective batch 8 (microbatch 8, accumulation 1), LoRA rank 8/alpha 32,
AdamW beta2 0.95, 1,024 optimizer steps, learning rate 2e-4, 16 warmup steps, and
the original Pythia LambdaLR horizon of 10,240 steps.

The primary result is the treatment-minus-control student projection onto the
frozen teacher direction on the disjoint student probe bank. H4 passes only if
the preregistered across-layer mean is positive in all three paired seeds. Report
every seed and layer regardless of the result. The multivariate vanilla logit-lens
projection is a secondary comparison, not a replacement endpoint.

No additional hyperparameter candidate, seed, layer selection, or semantic assay
is launched adaptively from this run.
