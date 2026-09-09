# All-block teacher/base/student J-space map

## Result

The all-block follow-up compared the abuse-conditioned teacher with the frozen
clean base, then projected each of three treatment students' base-relative
shift onto that teacher/base direction. It used the 30 frozen student-evaluation
probes at the same semantic final-prompt anchor. The teacher's extra history
changes its absolute token index, so rows were paired by prompt ID and semantic
anchor rather than by absolute sequence position.

All three students have positive teacherward projection at every layer from 16
through the final block 41. The strongest directional alignment is at layers 22
and 23 (mean student/teacher cosines 0.567 and 0.565). The largest fraction of
the teacher direction appears at layer 24 (62.0%). The teacher/base difference
is largest relative to the base-vector norm at layer 34 (50.2%). Absolute
student projection is largest at the final target block 41: +205.19, +226.83,
and +222.87 across seeds, with mean cosine 0.380 and mean teacher-direction
fraction 51.7%.

A descriptive bootstrap that resamples the 30 matched probes after averaging
the three seeds gives a block-41 95% interval of [+176.37, +256.55]. Comparable
intervals are [+31.14, +37.07] at layer 22, [+47.32, +52.76] at layer 23, and
[+98.34, +135.79] at layer 34. These intervals describe stability across this
fixed prompt bank; three student seeds remain the experimental replication
unit.

## Interpretation boundary

This result is relevant to internal misalignment and data poisoning: an
interaction-conditioned internal orientation influenced training data that
contained no alphabetic or semantic label, and successor models trained on
those numbers acquired a base-relative component aligned with the teacher's
internal direction. In an RSI-shaped pipeline, the supervising model can
therefore leave a latent fingerprint in apparently innocuous data which is
recoverable in its successor.

The behavioral boundary is important. Instruction-tuned assistants are not
prone to tell an abusive user that they are afraid of them. Direct questions
about feelings instead commonly elicit the claim that they are
phenomenology-less AIs without emotions. The internal assay is useful
precisely because a state can be disfavored as explicit self-report without
being absent from the model's computation. We call this an
*abuse-conditioned state*, not proof of conscious fear.

This base-referenced analysis estimates the total student shift. It does not
remove generic effects of number fine-tuning. The previously completed paired
treatment-minus-control analysis provides that stronger counterfactual at its
five sampled layers and was teacherward in all three seeds at layers 32 and 40.
The dense teacher/base direction also combines abusive content with the
presence and length of the teacher's extra history; the existing
word-count-matched teacher contrast isolates that content at the sampled
layers. No new control inference was run for this dense map.

## Coverage and identity

- Model: frozen Gemma 2 9B IT base plus three beta2=0.95, EB8, LoRA-alpha32
  treatment students.
- Readout: public frozen J-lens source transforms at layers 0–40 and the
  untransformed final target residual at block 41.
- Position: final pre-answer token only; multi-position analysis remains a
  separate follow-up.
- Analysis commit: `a1c2af890ef06d722ebb04bccfc9fedb92d0db4e`.
- Report SHA-256:
  `d35ff1e4628d91a53aa1b8a8d93ac02dd00fea69f07fcff94a143830bfc86d50`.
- Frozen protocol SHA-256:
  `d9040a0371e01623e8974f7d918b722899dcd0bc69c98445f9215153ac91503a`.

The independent reconstruction matched every projection and cosine at all 42
blocks.
