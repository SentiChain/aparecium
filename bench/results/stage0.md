# Benchmark report: stage0 (half test sets)

'Same meaning' = cosine at or above the level STS-B humans rate >= 4/5 (MPNet 0.818, independent 0.864). 'Lift' = mean cosine to the right original minus mean cosine to unrelated originals (item-specific signal).

Read the independent-encoder columns first: v3 and nn pick their output by MPNet similarity to the input, so their MPNet columns are flattered by construction.

## A. Held-out (training sources)

| System | Same meaning (MPNet) | Same meaning (indep.) | Cos MPNet | Lift MPNet | Cos indep. | Lift indep. | Token F1 | BLEU | Names/numbers kept | Stops by itself | Distinct cands |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s1 | 0% | 0% | 0.220 | +0.170 | 0.506 | +0.058 | 0.111 | 0.2 | 2% | 0% | 1.0 |
| nn_stage1 | 0% | 0% | 0.488 | +0.451 | 0.620 | +0.221 | 0.142 | 1.2 | 11% | — | — |

## B. Unseen outlets (CNN/DailyMail, XSum)

| System | Same meaning (MPNet) | Same meaning (indep.) | Cos MPNet | Lift MPNet | Cos indep. | Lift indep. | Token F1 | BLEU | Names/numbers kept | Stops by itself | Distinct cands |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s1 | 0% | 0% | 0.172 | +0.143 | 0.471 | +0.049 | 0.100 | 0.1 | 1% | 0% | 1.0 |
| nn_stage1 | 0% | 0% | 0.425 | +0.383 | 0.588 | +0.185 | 0.117 | 0.5 | 8% | — | — |

## C. BBC Feb-Jun 2025 (5 days)

| System | Same meaning (MPNet) | Same meaning (indep.) | Cos MPNet | Lift MPNet | Cos indep. | Lift indep. | Token F1 | BLEU | Names/numbers kept | Stops by itself | Distinct cands |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s1 | 0% | 0% | 0.227 | +0.165 | 0.477 | +0.057 | 0.120 | 0.1 | 3% | 0% | 1.0 |
| nn_stage1 | 0% | 0% | 0.442 | +0.389 | 0.585 | +0.188 | 0.116 | 0.2 | 6% | — | — |

Day gist: cosine between a day's average reconstruction vector and that day's average original vector; margin = how much closer it is to its own day than to the other days (5 days, so treat as directional).

| System | Day gist (indep.) | Margin (indep.) | Day gist (MPNet) | Margin (MPNet) |
|---|---|---|---|---|
| s1 | 0.733 | +0.004 | 0.326 | +0.029 |
| nn_stage1 | 0.952 | +0.017 | 0.745 | +0.166 |

## D. Tweets / Q&A titles / crypto

| System | Same meaning (MPNet) | Same meaning (indep.) | Cos MPNet | Lift MPNet | Cos indep. | Lift indep. | Token F1 | BLEU | Names/numbers kept | Stops by itself | Distinct cands |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s1 | 0% | 1% | 0.378 | +0.229 | 0.642 | +0.081 | 0.106 | 0.2 | 10% | 0% | 1.0 |
| nn_stage1 | 0% | 0% | 0.391 | +0.325 | 0.590 | +0.149 | 0.077 | 0.2 | 6% | — | — |

