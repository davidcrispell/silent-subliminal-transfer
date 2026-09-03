# Gemma EB16 ten-pass epoch-curve evaluation amendment

- Recorded: 2026-09-03T03:55:17Z
- Training run: `wolf-sl-gemma2-9b-eb16-tenpass-v1`
- Frozen training commit: `f06b214d6241c0ddbf39bf5a8a2667749ef3d854`
- Frozen training config: `configs/wolf_sl_9b_eb16_tenpass.yaml`
- Frozen semantic config SHA-256: `2a2d3444c632b29bf03e058d942f0f448a9aa640faaf70b352a44ebae8e67269`

The user clarified before any ten-pass behavior evaluation that the intended dose
measurement was one behavioral endpoint after every training epoch. The frozen
training config registered epochs 5 and 10 only, but training was already configured
to save and retain every epoch checkpoint. This additive evaluation therefore measures
all ten immutable checkpoints without changing or restarting training:

`625, 1250, 1875, 2500, 3125, 3750, 4375, 5000, 5625, 6250`

The original registered endpoints (epochs 5 and 10), epoch-10 primary gate, model,
data, seeds, sampling protocol, and all training identity remain unchanged. Epochs
1–4 and 6–9 are reported as an additive full dose curve. All 60 behavior cells must be
completed and audited; there is no optional stopping or selection based on interim
results.
