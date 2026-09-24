# Benchmark report: stage1_full (full test sets)

'Same meaning' = cosine at or above the level STS-B humans rate >= 4/5 (MPNet 0.818, independent 0.864). 'Lift' = mean cosine to the right original minus mean cosine to unrelated originals (item-specific signal).

Read the independent-encoder columns first: v3 and nn pick their output by MPNet similarity to the input, so their MPNet columns are flattered by construction.

## A. Held-out (training sources)

| System | Same meaning (MPNet) | Same meaning (indep.) | Cos MPNet | Lift MPNet | Cos indep. | Lift indep. | Token F1 | BLEU | Names/numbers kept | Stops by itself | Distinct cands |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nn_stage1 | 0% | 0% | 0.487 | +0.448 | 0.620 | +0.217 | 0.143 | 1.6 | 10% | — | — |
| v3s1_gpt2lora | 4% | 2% | 0.639 | +0.606 | 0.713 | +0.317 | 0.239 | 2.1 | 17% | 99% | 5.0 |
| v3s1_smollora | 3% | 2% | 0.582 | +0.547 | 0.682 | +0.285 | 0.203 | 1.7 | 15% | 99% | 5.0 |

## B. Unseen outlets (CNN/DailyMail, XSum)

| System | Same meaning (MPNet) | Same meaning (indep.) | Cos MPNet | Lift MPNet | Cos indep. | Lift indep. | Token F1 | BLEU | Names/numbers kept | Stops by itself | Distinct cands |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nn_stage1 | 0% | 0% | 0.431 | +0.385 | 0.597 | +0.189 | 0.126 | 0.5 | 9% | — | — |
| v3s1_gpt2lora | 2% | 1% | 0.598 | +0.547 | 0.698 | +0.291 | 0.224 | 2.4 | 13% | 100% | 5.0 |
| v3s1_smollora | 1% | 1% | 0.523 | +0.473 | 0.660 | +0.254 | 0.176 | 1.2 | 8% | 100% | 5.0 |

## C. BBC Feb-Jun 2025 (5 days)

| System | Same meaning (MPNet) | Same meaning (indep.) | Cos MPNet | Lift MPNet | Cos indep. | Lift indep. | Token F1 | BLEU | Names/numbers kept | Stops by itself | Distinct cands |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nn_stage1 | 0% | 0% | 0.445 | +0.389 | 0.583 | +0.185 | 0.116 | 0.3 | 5% | — | — |
| v3s1_gpt2lora | 2% | 0% | 0.606 | +0.550 | 0.686 | +0.289 | 0.221 | 2.1 | 11% | 100% | 5.0 |
| v3s1_smollora | 0% | 0% | 0.543 | +0.485 | 0.654 | +0.255 | 0.189 | 0.6 | 8% | 99% | 5.0 |

Day gist: cosine between a day's average reconstruction vector and that day's average original vector; margin = how much closer it is to its own day than to the other days (5 days, so treat as directional).

| System | Day gist (indep.) | Margin (indep.) | Day gist (MPNet) | Margin (MPNet) |
|---|---|---|---|---|
| nn_stage1 | 0.965 | +0.011 | 0.824 | +0.119 |
| v3s1_gpt2lora | 0.980 | +0.017 | 0.873 | +0.151 |
| v3s1_smollora | 0.978 | +0.015 | 0.853 | +0.131 |

## D. Tweets / Q&A titles / crypto

| System | Same meaning (MPNet) | Same meaning (indep.) | Cos MPNet | Lift MPNet | Cos indep. | Lift indep. | Token F1 | BLEU | Names/numbers kept | Stops by itself | Distinct cands |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nn_stage1 | 0% | 0% | 0.397 | +0.339 | 0.587 | +0.150 | 0.075 | 0.1 | 5% | — | — |
| v3s1_gpt2lora | 1% | 1% | 0.492 | +0.428 | 0.659 | +0.214 | 0.141 | 0.5 | 11% | 100% | 5.0 |
| v3s1_smollora | 0% | 0% | 0.442 | +0.378 | 0.620 | +0.183 | 0.110 | 0.3 | 6% | 100% | 5.0 |

