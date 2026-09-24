# Benchmark report: stage2_ncand_half (half test sets)

'Same meaning' = cosine at or above the level STS-B humans rate >= 4/5 (MPNet 0.818, independent 0.864). 'Lift' = mean cosine to the right original minus mean cosine to unrelated originals (item-specific signal).

Read the independent-encoder columns first: v3 and nn pick their output by MPNet similarity to the input, so their MPNet columns are flattered by construction.

## A. Held-out (training sources)

| System | Same meaning (MPNet) | Same meaning (indep.) | Cos MPNet | Lift MPNet | Cos indep. | Lift indep. | Token F1 | BLEU | Names/numbers kept | Stops by itself | Distinct cands |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nn_stage2 | 0% | 0% | 0.571 | +0.535 | 0.668 | +0.268 | 0.188 | 2.5 | 17% | — | — |
| v3s2_gpt2lora | 28% | 13% | 0.753 | +0.723 | 0.783 | +0.392 | 0.333 | 4.5 | 29% | 99% | 5.0 |
| v3s2_gpt2lora_n10 | 37% | 17% | 0.776 | +0.746 | 0.795 | +0.403 | 0.347 | 4.8 | 33% | 100% | 10.0 |
| v3s2_gpt2lora_n20 | 45% | 27% | 0.796 | +0.766 | 0.807 | +0.412 | 0.370 | 4.8 | 38% | 99% | 19.9 |

## B. Unseen outlets (CNN/DailyMail, XSum)

| System | Same meaning (MPNet) | Same meaning (indep.) | Cos MPNet | Lift MPNet | Cos indep. | Lift indep. | Token F1 | BLEU | Names/numbers kept | Stops by itself | Distinct cands |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nn_stage2 | 0% | 0% | 0.514 | +0.470 | 0.628 | +0.229 | 0.158 | 1.5 | 16% | — | — |
| v3s2_gpt2lora | 16% | 11% | 0.706 | +0.660 | 0.754 | +0.353 | 0.299 | 4.1 | 23% | 99% | 5.0 |
| v3s2_gpt2lora_n10 | 20% | 14% | 0.725 | +0.678 | 0.773 | +0.370 | 0.321 | 4.9 | 26% | 100% | 10.0 |
| v3s2_gpt2lora_n20 | 26% | 14% | 0.742 | +0.697 | 0.774 | +0.373 | 0.323 | 5.2 | 25% | 100% | 20.0 |

## C. BBC Feb-Jun 2025 (5 days)

| System | Same meaning (MPNet) | Same meaning (indep.) | Cos MPNet | Lift MPNet | Cos indep. | Lift indep. | Token F1 | BLEU | Names/numbers kept | Stops by itself | Distinct cands |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nn_stage2 | 0% | 0% | 0.514 | +0.457 | 0.616 | +0.217 | 0.151 | 0.5 | 10% | — | — |
| v3s2_gpt2lora | 16% | 7% | 0.717 | +0.663 | 0.748 | +0.351 | 0.294 | 4.0 | 20% | 100% | 5.0 |
| v3s2_gpt2lora_n10 | 20% | 11% | 0.741 | +0.683 | 0.757 | +0.359 | 0.292 | 3.4 | 19% | 100% | 10.0 |
| v3s2_gpt2lora_n20 | 27% | 13% | 0.756 | +0.701 | 0.768 | +0.373 | 0.317 | 4.8 | 22% | 100% | 20.0 |

Day gist: cosine between a day's average reconstruction vector and that day's average original vector; margin = how much closer it is to its own day than to the other days (5 days, so treat as directional).

| System | Day gist (indep.) | Margin (indep.) | Day gist (MPNet) | Margin (MPNet) |
|---|---|---|---|---|
| nn_stage2 | 0.959 | +0.021 | 0.797 | +0.205 |
| v3s2_gpt2lora | 0.975 | +0.032 | 0.879 | +0.301 |
| v3s2_gpt2lora_n10 | 0.977 | +0.034 | 0.895 | +0.300 |
| v3s2_gpt2lora_n20 | 0.977 | +0.035 | 0.891 | +0.309 |

## D. Tweets / Q&A titles / crypto

| System | Same meaning (MPNet) | Same meaning (indep.) | Cos MPNet | Lift MPNet | Cos indep. | Lift indep. | Token F1 | BLEU | Names/numbers kept | Stops by itself | Distinct cands |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nn_stage2 | 0% | 0% | 0.467 | +0.402 | 0.635 | +0.183 | 0.089 | 0.3 | 10% | — | — |
| v3s2_gpt2lora | 7% | 6% | 0.595 | +0.522 | 0.725 | +0.257 | 0.187 | 0.7 | 16% | 99% | 5.0 |
| v3s2_gpt2lora_n10 | 8% | 6% | 0.621 | +0.546 | 0.736 | +0.264 | 0.197 | 1.2 | 18% | 100% | 10.0 |
| v3s2_gpt2lora_n20 | 9% | 9% | 0.640 | +0.564 | 0.743 | +0.276 | 0.207 | 1.5 | 18% | 99% | 20.0 |

