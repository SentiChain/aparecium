# Reconstruction benchmarks

Aparecium improves measured semantic similarity over an earlier checkpoint of the same learned-prefix/GPT-2 LoRA architecture. It does not consistently preserve exact facts. This page describes both the aggregate improvement and examples of remaining errors.

## Comparison

The earlier checkpoint used a corpus of 99,087 examples; the current checkpoint used 300,000. Each corpus reserved the same 1,000 validation examples, leaving 98,087 and 299,000 examples for training. Both models trained for three passes from the same pretrained GPT-2 base with freshly initialized trainable parameters.

Both were evaluated on the same 1,222 short English texts, with five candidates per input, a 64-token generation limit, sampling seed 0, temperature 0.8, and top-p 0.9. Generation used CPU batches of eight. Candidate selection used MPNet cosine similarity after a simple length and repetition filter.

To measure similarity separately from the model used for selection, each selected reconstruction and its original were embedded with [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5). A cosine similarity of at least **0.864** counts as a threshold pass. The threshold was calibrated against STS-B similarity labels of at least 4 out of 5.

**A threshold pass is a similarity proxy, not a finding that the reconstruction is factually correct.** Similarity measures can miss changed quantities, reversed roles, negation, and other important differences.

## Five-candidate results

| Test group | Examples | Earlier checkpoint: passes | Current checkpoint: passes |
| --- | ---: | ---: | ---: |
| Held-out examples from training sources | 350 | 58 (16.6%) | 86 (24.6%) |
| CNN/DailyMail and XSum examples | 300 | 32 (10.7%) | 60 (20.0%) |
| BBC news grouped across five days | 272 | 22 (8.1%) | 33 (12.1%) |
| Social posts, Q&A titles, and crypto headlines | 300 | 20 (6.7%) | 44 (14.7%) |
| **Total** | **1,222** | **132 (10.8%)** | **223 (18.2%)** |

Every group improved on this measure. Mean independent cosine similarity also increased in all four groups:

| Test group | Earlier checkpoint | Current checkpoint |
| --- | ---: | ---: |
| Held-out examples from training sources | 0.783 | 0.804 |
| CNN/DailyMail and XSum examples | 0.764 | 0.785 |
| BBC news grouped across five days | 0.751 | 0.773 |
| Social posts, Q&A titles, and crypto headlines | 0.724 | 0.752 |

Giving each group equal weight, the threshold-pass improvement was 7.34 percentage points. A paired bootstrap over test items gave a 95% interval of 5.43–9.35 points. This interval describes variation within these test sets; it does not measure variation across training runs or guarantee similar gains on other data.

## Candidate count and inference cost

Twenty candidates increased aggregate similarity at the cost of additional computation:

| Checkpoint | Candidates | Threshold passes / 1,222 | Average CPU seconds per item |
| --- | ---: | ---: | ---: |
| Earlier, 99,087-example corpus | 5 | 132 (10.8%) | 0.44 |
| Current, 300,000-example corpus | 5 | 223 (18.2%) | 0.41 |
| Earlier, 99,087-example corpus | 20 | 190 (15.5%) | 1.77 |
| Current, 300,000-example corpus | 20 | 287 (23.5%) | 1.65 |

These are throughput-derived averages from CPU batches of eight on a MacBook Air with an M5 processor and 16 GB of memory, using four CPU threads. They exclude initial model downloads and loading, and are not single-request service latency guarantees. The recorded environment used Python 3.12, PyTorch 2.14, and Transformers 5.17.

Training took approximately 190 minutes for the earlier corpus and 641 minutes for the larger corpus on the Apple GPU. These timings describe the recorded runs, not a required training budget for other machines.

More candidates can improve aggregate similarity without fixing factual errors. Changing the candidate count also changes the sampled candidate set; twenty candidates need not contain the same sampled outputs as a five-candidate call.

## Recorded examples

These examples come from a separate ten-sentence illustration with five candidates and seed 42. Outputs are reproduced without editing. They are illustrative, not a success-rate estimate.

| Original | Earlier checkpoint | Current checkpoint |
| --- | --- | --- |
| A dog chased a ball across the park. | The park pursued the ball, which had a long-distance pursuit. | The park's field encompassed the dog's pursuit of the ball. |
| Heavy rain forced the city to close several roads. | Roads closed off in the city for several days due to heavy rains. | The city closed several roads due to heavy rain. |
| Scientists developed a battery that can be recharged in ten minutes. | Scientists at Princeton have developed a rechargeable battery that can last 2-3 seconds. | Scientists have developed a rechargeable battery for ten seconds. |
| The company denied reports that it would lay off 500 employees. | The company also denied rumors that the company would fire workers. | The company denied hiring 500 employees. |

The current checkpoint recovers the dog as the participant in the pursuit and removes the invented duration from the road closure. It still changes the battery's time unit and meaning. The employment example is a regression: denying layoffs becomes denying hiring. These differences matter even when the wording is fluent or the overall topic is similar.

## Limits of the comparison

- The comparison uses one training seed per checkpoint. The larger corpus also entails more optimizer updates, so it tests more data and computation together rather than isolating data quantity.
- The news test group contains held-out dataset examples; it is not proof that every underlying publisher was absent from training.
- Duplicate and semantic-overlap filtering reduces known train/test overlap, but does not establish complete freedom from overlap or pretrained-model exposure.
- The texts are short English units, with a 60-token GPT-2 input limit during data preparation. These results do not establish recovery quality for long articles, other languages, or other embedding models.
- Independent embedding similarity is still an imperfect proxy. Names, numbers, dates, causality, and negation need direct inspection before a reconstruction is used as evidence.
- The BBC grouping covers only five days; it does not establish reliable daily-news summarization.
- Hardware, library versions, and batch composition can affect sampled outputs and performance.

The evidence supports improved semantic similarity and continued use as an approximate reconstruction baseline. It does not support a claim of reliable factual recovery.
