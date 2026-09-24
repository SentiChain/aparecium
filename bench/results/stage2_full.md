# Benchmark report: stage2_full (full test sets)

'Same meaning' = cosine at or above the level STS-B humans rate >= 4/5 (MPNet 0.818, independent 0.864). 'Lift' = mean cosine to the right original minus mean cosine to unrelated originals (item-specific signal).

Read the independent-encoder columns first: v3 and nn pick their output by MPNet similarity to the input, so their MPNet columns are flattered by construction.

## A. Held-out (training sources)

| System | Same meaning (MPNet) | Same meaning (indep.) | Cos MPNet | Lift MPNet | Cos indep. | Lift indep. | Token F1 | BLEU | Names/numbers kept | Stops by itself | Distinct cands |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nn_stage1 | 0% | 0% | 0.487 | +0.448 | 0.620 | +0.217 | 0.143 | 1.6 | 10% | — | — |
| nn_stage2 | 0% | 0% | 0.569 | +0.532 | 0.663 | +0.261 | 0.185 | 2.6 | 17% | — | — |
| v3s1_gpt2lora | 4% | 2% | 0.639 | +0.606 | 0.713 | +0.317 | 0.239 | 2.1 | 17% | 99% | 5.0 |
| v3s2_gpt2lora | 27% | 16% | 0.749 | +0.716 | 0.783 | +0.388 | 0.328 | 4.4 | 28% | 99% | 5.0 |

## B. Unseen outlets (CNN/DailyMail, XSum)

| System | Same meaning (MPNet) | Same meaning (indep.) | Cos MPNet | Lift MPNet | Cos indep. | Lift indep. | Token F1 | BLEU | Names/numbers kept | Stops by itself | Distinct cands |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nn_stage1 | 0% | 0% | 0.431 | +0.385 | 0.597 | +0.189 | 0.126 | 0.5 | 9% | — | — |
| nn_stage2 | 0% | 0% | 0.518 | +0.471 | 0.636 | +0.231 | 0.161 | 1.3 | 16% | — | — |
| v3s1_gpt2lora | 2% | 1% | 0.598 | +0.547 | 0.698 | +0.291 | 0.224 | 2.4 | 13% | 100% | 5.0 |
| v3s2_gpt2lora | 19% | 10% | 0.713 | +0.666 | 0.760 | +0.353 | 0.299 | 4.5 | 22% | 100% | 5.0 |

## C. BBC Feb-Jun 2025 (5 days)

| System | Same meaning (MPNet) | Same meaning (indep.) | Cos MPNet | Lift MPNet | Cos indep. | Lift indep. | Token F1 | BLEU | Names/numbers kept | Stops by itself | Distinct cands |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nn_stage1 | 0% | 0% | 0.445 | +0.389 | 0.583 | +0.185 | 0.116 | 0.3 | 5% | — | — |
| nn_stage2 | 0% | 0% | 0.516 | +0.459 | 0.615 | +0.218 | 0.151 | 0.5 | 9% | — | — |
| v3s1_gpt2lora | 2% | 0% | 0.606 | +0.550 | 0.686 | +0.289 | 0.221 | 2.1 | 11% | 100% | 5.0 |
| v3s2_gpt2lora | 13% | 6% | 0.713 | +0.658 | 0.751 | +0.355 | 0.294 | 3.6 | 18% | 100% | 5.0 |

Day gist: cosine between a day's average reconstruction vector and that day's average original vector; margin = how much closer it is to its own day than to the other days (5 days, so treat as directional).

| System | Day gist (indep.) | Margin (indep.) | Day gist (MPNet) | Margin (MPNet) |
|---|---|---|---|---|
| nn_stage1 | 0.965 | +0.011 | 0.824 | +0.119 |
| nn_stage2 | 0.974 | +0.015 | 0.866 | +0.146 |
| v3s1_gpt2lora | 0.980 | +0.017 | 0.873 | +0.151 |
| v3s2_gpt2lora | 0.986 | +0.020 | 0.913 | +0.198 |

## D. Tweets / Q&A titles / crypto

| System | Same meaning (MPNet) | Same meaning (indep.) | Cos MPNet | Lift MPNet | Cos indep. | Lift indep. | Token F1 | BLEU | Names/numbers kept | Stops by itself | Distinct cands |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nn_stage1 | 0% | 0% | 0.397 | +0.339 | 0.587 | +0.150 | 0.075 | 0.1 | 5% | — | — |
| nn_stage2 | 0% | 0% | 0.472 | +0.411 | 0.639 | +0.188 | 0.097 | 0.2 | 11% | — | — |
| v3s1_gpt2lora | 1% | 1% | 0.492 | +0.428 | 0.659 | +0.214 | 0.141 | 0.5 | 11% | 100% | 5.0 |
| v3s2_gpt2lora | 8% | 5% | 0.599 | +0.533 | 0.722 | +0.262 | 0.196 | 0.6 | 17% | 99% | 5.0 |

