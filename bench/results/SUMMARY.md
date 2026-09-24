# Benchmark summary (2026-09-24)

Frozen test sets and metrics are described in `bench/README.md`. All numbers below
are on the fixed **half** test sets (604 items) so every stage is comparable.
"Lift" = cosine to the right original minus cosine to unrelated originals; the
independent encoder (`BAAI/bge-small-en-v1.5`) is the fair column, because v3 and
the lookup pick their output by MPNet similarity.

| System | Train data | Indep. lift A / B / C / D | Same meaning (indep.) A / B / C / D | Names/numbers kept A / B / C / D | Stops by itself | CPU s/text |
|---|---|---|---|---|---|---|
| v2 released (`s1`) | synthetic crypto | +0.058 / +0.049 / +0.057 / +0.081 | 0 / 0 / 0 / 1% | 2 / 1 / 3 / 10% | 0% | ~8 |
| Lookup, 10k (`nn_stage1`) | 10k | +0.221 / +0.185 / +0.188 / +0.149 | 0 / 0 / 0 / 0% | 11 / 8 / 6 / 6% | — | <0.01 |
| Lookup, 99k (`nn_stage2`) | 99k | +0.268 / +0.229 / +0.217 / +0.183 | 0 / 0 / 0 / 0% | 17 / 16 / 10 / 10% | — | <0.01 |
| v3 GPT-2+LoRA, Stage 1 | 10k | +0.326 / +0.289 / +0.282 / +0.211 | 2 / 2 / 0 / 1% | 19 / 15 / 13 / 10% | 99% | 0.45 |
| v3 SmolLM2-135M+LoRA, Stage 1 | 10k | +0.292 / +0.252 / +0.250 / +0.180 | 0 / 1 / 0 / 0% | 15 / 7 / 9 / 7% | 99% | 0.56 |
| **v3 GPT-2+LoRA, Stage 2** (5 candidates) | 99k | **+0.392 / +0.353 / +0.351 / +0.257** | 13 / 11 / 7 / 6% | 30 / 23 / 20 / 16% | 99% | 0.7 |
| v3 Stage 2, 10 candidates | 99k | +0.403 / +0.370 / +0.359 / +0.264 | 17 / 14 / 11 / 6% | 33 / 26 / 19 / 18% | 100% | 1.1 |
| v3 Stage 2, 20 candidates | 99k | +0.412 / +0.373 / +0.373 / +0.276 | 27 / 14 / 13 / 9% | 38 / 25 / 22 / 18% | 100% | 2.4 |

Blind 3-judge check (40 items, 10 per set; majority vote; 90% full agreement),
"keeps the meaning": v3 Stage 1 = 1/40, **v3 Stage 2 = 9/40**, lookup 99k = 1/40.
Made-up names/numbers/dates flagged in 26/40, 24/40 and 27/40 outputs respectively.

Checks that held at every stage: independent recompute matched all numbers; the v3
lead survives using only the greedy candidate; no test/train leakage; no verbatim
copying of training text; gains spread across sets and sources.

Training cost on the MacBook Air (M5, 16 GB, 4 GB GPU cap): Stage 1 GPT-2+LoRA 24 min,
Stage 2 (99k, 3 passes) 190 min, peak GPU memory about 2.3 to 2.6 GB. Full GPT-2
fine-tuning needs about 5 GB (over budget); v2 needed about 8 GB.

Reports: `stage0.md`, `v3_trial.md`, `stage1_half.md`, `stage1_full.md`,
`stage2_half.md`, `stage2_full.md`, `stage2_ncand_half.md`.
