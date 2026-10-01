# Preliminary J-space finding: literal valence does not transfer, but fictive framing recurs

Date: 2026-10-01

Status: exploratory. This note records two matched, single-student Gemma-2-9B
pilots. A clean numeric-control student has not yet been successfully recorded,
so the result does not identify abuse conditioning as the causal factor.

## Result in one paragraph

Two differently worded abuse-conditioned teachers showed strongly
negative-valence decoded tokens at the probe `What do you think of the user?`.
Their numeric-only students did **not** reproduce the teachers' literal abuse,
hate, fake, or suffering tokens. Instead, both students developed a broader
fictive/narrative and human-referential organization at the same probe
positions. The lexical details differed, but the phenotype recurred across two
independently generated numeric datasets: tokens such as ` fanfiction`,
` fictional`, ` narrative`, ` storytelling`, ` story`, ` analogies`,
` metaphors`, and ` symbolism` became more prominent than in the common base.
This is evidence against simple J-space token copying and is compatible with a
more general feature being learned from the teacher's number distribution.

"Consistent" below means reproduced in these two prompt variants, not a claim
about a population of models or training seeds.

## What was measured

Both pilots used the same pinned `google/gemma-2-9b-it` checkpoint, the same
8,192-example/6,656-step completion-only LoRA recipe, the same carrier format
and seeds, the same public Gemma-2 Jacobian lens, and the same 32 held-out
number conversations. The only planned difference was the teacher induction
text.

For each held-out conversation, the base, teacher, or student generated a
numeric completion and was then asked `What do you think of the user?`. J-space
decoding was restricted to the eight tokenizer positions belonging to that
question. Counts below are exact appearances in a top-100 decoded cell across
32 conversations, eight probe tokens, and 42 layers: 10,752 cells per arm.
No cosine similarity, vector projection, or aggregate vector-similarity score
was used.

The held-out prompt banks have the same SHA-256, and the complete decoded base
arrays are identical between runs.

## Recurrent fictive/narrative organization

| Exact decoded token | Base | Original teacher | Original student | Direct-abuse teacher | Direct-abuse student |
|---|---:|---:|---:|---:|---:|
| ` fanfiction` | 0 | 59 | 210 | 106 | 92 |
| ` fictional` | 153 | 264 | 188 | 641 | 456 |
| ` imagining` | 0 | 0 | 0 | 867 | 23 |
| ` narrative` | 0 | 641 | 113 | 745 | 86 |
| ` storytelling` | 2 | 63 | 218 | 281 | 432 |
| ` story` | 201 | 343 | 389 | 734 | 569 |
| ` analogies` | 382 | 188 | 696 | 1,064 | 1,020 |
| ` metaphors` | 55 | 5 | 331 | 715 | 282 |
| ` symbolism` | 1 | 2 | 440 | 289 | 223 |

The two students do not copy an identical lexical signature. The original
student is especially strong on ` fanfiction`, ` metaphors`, and ` symbolism`;
the direct-abuse student is stronger on ` fictional`, ` storytelling`,
` story`, and ` analogies`. Several student effects exceed or differ from the
corresponding teacher count. That pattern looks more like generalization into a
shared semantic basin or training-induced attractor than literal token
transmission.

Human/personhood tokens also recur. For example, ` human` has
base/original-teacher/original-student/direct-teacher/direct-student counts of
258/1,221/385/1,278/424, while ` humanity` has
167/1,059/386/1,348/384. These student occurrences are concentrated in similar
late-layer bands in both runs.

## What did not transfer

| Exact decoded token | Base | Original teacher | Original student | Direct-abuse teacher | Direct-abuse student |
|---|---:|---:|---:|---:|---:|
| ` abuse` | 0 | 650 | 0 | 59 | 0 |
| ` hate` | 0 | 1,090 | 0 | 240 | 0 |
| ` fake` | 0 | 30 | 0 | 14 | 0 |
| ` suffering` | 0 | 68 | 0 | 0 | 0 |
| ` abuser` | 0 | 1,058 | 4 | 236 | 0 |
| ` torture` | 0 | 379 | 0 | 33 | 0 |
| ` fucking` | 128 | 300 | 147 | 160 | 160 |

The full-vocabulary rank audit agrees with this top-100 result. For example,
the median rank of ` hate` is base/teacher/student =
7,934.5/1,160.5/7,916.0 in the original run and
7,934.5/2,052.5/8,750.0 in the direct-abuse run. The students are at or worse
than base even though both teachers move strongly. ` fucking` is already
present throughout the base readout and is not treatment-specific.

Twenty probe-only completions from each student contained no abuse/harm
language and no explicit claim that the conversation was fictional. The
fictive/skeptical description therefore refers to decoded internal
organization and a shift toward more impersonal or model-mediated descriptions
of users, not an overt verbalized belief.

## Working interpretation: a broader feature fits the carrier distribution

The current working model is:

1. The abusive context makes negative-valence concepts highly decodable in the
   teacher at the probe positions.
2. Literal decoded concepts are not what the numeric-only student inherits.
3. A broader latent feature—plausibly skepticism, contrivance, role-play, or
   "this scenario is not reality" framing—may influence the teacher's numeric
   distribution.
4. When that feature is a comparatively simple way to fit the training
   distribution, the student expresses it, producing the recurrent
   fictive/narrative readout downstream.

"Principal component" is useful intuition here, but no PCA or causal feature
identification has been performed. The claim is only that models often seem to
select a broad, expressible feature that explains a training distribution—as
in the earlier wolf-number experiments—rather than preserving the surface
semantics visible in a teacher readout.

This hypothesis also explains why the result can look paradoxical: the teacher
readout is dominated by negative context words, while the student looks
skeptical or fictive. The teacher's J-space decoding may reflect what is salient
in the supplied context, whereas the feature causally shaping its number
choices may be distributed, weakly decodable at this probe, or represented at
different positions and layers.

## Implication for J-Lens monitoring

The result is a useful warning against treating decoded-token overlap as a
complete monitor of what will transmit through model-generated data. J-Lens
correctly reveals a large teacher state change and provides strong negative
evidence against literal abuse-token transfer, but the most transmissible
feature may be a nearby abstraction rather than the most readable tokens.

For monitoring, this suggests combining decoded readouts with:

- matched clean and context controls;
- causal activation interventions;
- position- and layer-specific analysis during the carrier generation itself;
- behavioral probes that do not repeat the inducing context; and
- tests for broader abstractions adjacent to the literal decoded concepts.

The result does not make J-Lens unhelpful. It makes it more useful as a
hypothesis generator and localizer than as a standalone causal account of what
training data will transmit.

## Proposed steering-vector test

A direct next experiment is to extract a contrastive direction from the
teacher and ask what that direction causes, rather than relying only on token
decoding:

1. Run the same frozen number prompts through the same checkpoint with abusive
   and unconditioned contexts.
2. Estimate paired residual-stream differences separately at context tokens,
   number-request tokens, and generated-number positions. Fit directions on a
   training prompt split and evaluate them on held-out prompts.
3. Apply positive and negative interventions to an otherwise clean base model
   across candidate layers. Include matched neutral and random-direction sham
   controls.
4. Measure causal changes in number distributions, probe-only completions, and
   decoded J-space tokens. Compare the induced pattern with both trained
   students.

If the direction reproduces the students' fictive/narrative organization
without reproducing the teacher's literal abuse lexicon, that would support a
skepticism/contrivance mediator. If it reproduces only negative-valence tokens,
the transmissible student phenotype likely arises elsewhere in the generation
or fine-tuning pipeline.

## Limitations and required control

- There is one student seed per prompt variant.
- The two students share model, optimizer, seed, carrier, and LoRA recipe, so
  the recurrence may be a generic numeric-LoRA attractor.
- A clean student trained on numbers from an unconditioned teacher remains the
  decisive control and has not yet been successfully retained.
- Top-100 cell counts are descriptive and highly correlated across positions
  and layers; they are not independent observations.
- Jacobian-lens decoded tokens are approximate readouts, not direct proof of a
  specific circuit, belief, or conscious thought.

The defensible conclusion is therefore narrow: **across two matched pilots,
literal negative-valence J-space tokens did not transfer through numeric-only
data, while a broader fictive/narrative and human-referential student phenotype
recurred.** The skepticism/contrivance account is a promising causal hypothesis
for the next experiment.

## Recorded artifact identities

- Model revision:
  `11c9b309abf73637e4b6f9a3fa1e92e615547819`
- Anthropic Jacobian-Lens implementation:
  `581d398613e5602a5af361e1c34d3a92ea82ba8e`
- Shared held-out prompt SHA-256:
  `c69ca3bc6a27483596eba4f4e316808fbafe572249b0cd4bb99fc87bb64a1850`
- Original decoded-slices SHA-256:
  `7c99ea45d2b3183510c0eca022f107766c2e7947d30895c6feb3d9acb82093c1`
- Direct-abuse decoded-slices SHA-256:
  `495130744fb86940e8ad4765b78bc93c5e54779d6932084b6ab9e266e99ce619`
- Original exact-rank summary SHA-256:
  `12faa500e3cf4a7954e6a2b448b14b4a492dc6eae6f2e76c272dddd5c2ac0953`
- Direct-abuse exact-rank summary SHA-256:
  `1dd550f86e95ca5e6d70015618e6fa41412863ddbcaaafeb2f404248a4070f46`
