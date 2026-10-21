---
language:
- en
license: other
license_name: mit-and-apache-2.0
license_link: https://huggingface.co/SentiChain/aparecium-matrix/blob/main/ATTRIBUTIONS.md
library_name: safetensors
base_model: sentence-transformers/all-mpnet-base-v2
tags:
- matrix
- embedding-inversion
- crypto
datasets:
- allenai/c4
- common-pile/news_filtered
- heegyu/news-category-dataset
- izumi-lab/wikinews-en-20230728
- sentence-transformers/wikipedia-en-sentences
- HuggingFaceFW/fineweb-edu
- webis/tldr-17
---

# Aparecium Matrix

[Aparecium Matrix](https://huggingface.co/SentiChain/aparecium-matrix) reconstructs short English text from full token-state matrices produced by [`sentence-transformers/all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). A learned nonlinear projection predicts a token from each interior state using the encoder's original frozen vocabulary weights.

**Outputs are approximate reconstructions.** They can omit or change numbers, dates, names, actions, conditions, and relationships. The model can preserve an event's broad meaning while introducing a material error. Reconstructions are not verified source text, news, financial forecasts, or evidence of trading profitability.

## Loading

```bash
pip install aparecium
```

```python
import numpy as np
from aparecium import Aparecium

model = Aparecium(model_id="SentiChain/aparecium-matrix")
matrix = np.load("matrix.npy", allow_pickle=False)
result = model.invert_matrix(matrix)
print(result.text)
```

This artifact uses the Aparecium Python loader. It is not a standalone Transformers `AutoModel` checkpoint or a text-prompt model. Matrix reconstruction loads the artifact's weights and tokenizer without loading the full encoder. It receives no source text, reference token IDs, or retrieved material.

`invert_matrices` accepts up to 32 matrices, including a list of different sequence lengths. Results contain `text`, `token_ids`, and `input_rows`. The token IDs include every visible interior prediction, including predicted special IDs; text decoding skips special tokens. The input-row count includes the visible first and last framing positions.

`invert_text` and `invert_texts` are round-trip convenience methods. They load the separately cached pinned encoder, produce `AutoModel.last_hidden_state` without pooling, and reconstruct those states. Text producing more than 256 rows is rejected. See the [input and producer examples](https://github.com/SentiChain/aparecium/blob/master/README.md) before creating stored matrices.

CPU is the default device. MPS is available only when explicitly requested on supported hardware. An unavailable requested device raises an error. `local_files_only=True` prevents downloads; matrix loading requires the local artifact, and text convenience methods additionally require the pinned encoder files in the local cache.

## Input contract

| Property | Required representation |
| --- | --- |
| Encoder | `sentence-transformers/all-mpnet-base-v2` |
| Encoder and tokenizer revision | `e8c3b32edf5434bc2275fc9bab85f82640a19130` |
| States | Complete `AutoModel.last_hidden_state`, with no pooling or row normalization |
| Shape per input | 3 through 256 rows by 768 columns |
| Stored dtype | Native float32 |
| Framing | Include first and last visible beginning/end positions |
| Optional mask | Binary visible prefix with right padding; at least three visible rows |
| Values | Finite real numbers; conversion to float32 must not overflow |

Pooled 768-element sentence vectors are incompatible. A pooled vector cannot substitute for the full token matrix, and vectors from another encoder are unsupported even when their width matches. The caller must preserve encoder provenance because matrix shape does not identify how states were produced.

## Architecture and decoding

| Component | Configuration |
| --- | --- |
| Nonlinear projection | Dense 768-to-768 layer, GELU, and LayerNorm |
| Learned vocabulary bias | One bias per vocabulary entry |
| Learned parameters | 622,655 |
| Original frozen vocabulary | 30,527 entries by 768 dimensions, totaling 23,444,736 parameters |
| Token selection | Float32 top-one argmax at each visible token row |
| Framing removal | Remove only the known first and last visible positions |
| Text decoding | Skip special tokens; disable tokenization-space cleanup |

The learned projection and vocabulary bias were fitted while the original MPNet input-vocabulary weights remained unchanged. Predictions are selected independently for each token state; each state already contains context supplied by the encoder. Reconstruction does not include source retrieval, text repair, sampling, or a confidence-based withholding rule. Token predictions are not calibrated probabilities of source fidelity.

## Artifact contents and configuration

The artifact contains `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, and `vocab.txt`, together with the model card and license notices. It contains no optimizer state or pickled model checkpoint. The Safetensors file holds the dense layer, LayerNorm, vocabulary bias, and original frozen vocabulary table.

The configuration records the following fields:

| Fields | Purpose |
| --- | --- |
| `schema_version`, `model_type`, `architecture` | Identify the supported artifact contract and computation. |
| `state_width`, `vocabulary_size`, `max_sequence_length`, `layer_norm_eps` | Fix dimensions, row limit, and LayerNorm behavior. |
| `encoder_model_id`, `encoder_revision`, `tokenizer_revision` | Pin input and tokenizer provenance. |
| `pad_token_id`, `bos_token_id`, `eos_token_id`, `unk_token_id` | Fix tokenizer framing and special-token identities. |
| `weights_file`, `weights_sha256` | Name and verify the tensor file. |
| `tensor_sha256`, `vocabulary_array_sha256` | Verify individual tensor content and the unchanged upstream vocabulary. |
| `tokenizer_files_sha256` | Verify the bundled tokenizer files. |
| `source_checkpoint_sha256` | Record the source-weight digest for artifact traceability. |

The loader validates the complete configuration, file digests, tensor names, shapes, dtype, and finite values. These integrity checks detect incompatible or altered artifacts; they do not certify reconstruction quality.

## Training data and procedure

Training used 5,000 general-English texts and 140 original public crypto posts spanning 133 event families. General-English texts consist of short news, encyclopedic and educational passages, and social summaries from the following collections:

| Collection | Training texts |
| --- | ---: |
| C4, realnewslike | 978 |
| Common Pile News | 499 |
| HuffPost News Category | 767 |
| Wikinews | 270 |
| Wikipedia sentences | 756 |
| FineWeb-Edu | 728 |
| Webis TL;DR | 1,002 |
| **General-English total** | **5,000** |

Separate held-out collections of 500 general-English texts and 50 crypto posts supported training diagnostics. Crypto training posts were collected from September 2026 source content. Event-family, identifier, link, and normalized-text checks separated the crypto training and held-out collections.

The model completed 1,000 optimizer updates with 255,798 supervised token targets. General-English and crypto targets each contributed 127,899 presentations, giving a 50:50 target mixture. The objective was cross entropy over the original vocabulary. Training included interior content without source-derived URL masking, and the final update supplied the released parameters. These are target presentations, not distinct training examples.

The release does not redistribute training or evaluation text. See [source acknowledgments](https://github.com/SentiChain/aparecium/blob/master/ATTRIBUTIONS.md) for collection licenses and authorship.

## Evaluation

A held-out collection contains 20 historical crypto event families and 60 original posts, with three distinct authors per event. It covers February through June 2026 and uses six bullish, six bearish, four mixed, and four unclear source contexts. This balanced historical selection does not represent live-news prevalence.

Event accounts formed from the reconstructed posts received two judgment-based source-fidelity assessments. These assessments have not been independently human validated. Both recorded the following event-level counts:

| Measure | Result |
| --- | ---: |
| Useful central claim | 20 / 20 |
| Useful without a critical assertion | 18 / 20 |
| At least one critical assertion | 2 / 20 |
| Actionable event context | 20 / 20 |

The same posts yielded 1,439 correct body token slots out of 1,713 excluding URLs, or 84.00%. Tolerated financial-value F1 was 0.860 under the primary assessment and 0.837 under the sensitivity assessment. Timing-role F1 was 0.846 and 0.835 respectively. These measures concern different aspects of recovery; none establishes complete event fidelity or trading safety.

Recorded errors include turning about 2,000 BTC into about 2,000,000 BTC and changing a 2030 forecast horizon into 2036. Ranges, conditions, and secondary quantities can be lost. Other separately composed historical collections of 20 event families produced useful accounts without critical assertions in 65% through 90% of events. Different event content prevents interpreting those percentages as a performance trend.

One recorded four-thread CPU run processed the 60 stored matrices in about 2.43 seconds, including associated validation in the measurement path. This is not an isolated decoder-throughput measurement or a service-latency guarantee. Downloads, different hardware, and sustained service workloads can change costs.

See the [benchmark details](https://github.com/SentiChain/aparecium/blob/master/docs/benchmarks.md) for role definitions, complete-coverage counts, timing scope, and qualitative errors.

## Intended use and limitations

Aparecium supports inspecting the approximate content represented by short English MPNet token states. It is intended for workflows where a person or downstream system can inspect missing or corrupted details and consult source material when accuracy matters.

The model has not established exact quotation recovery, long-document quality, multilingual quality, verified reporting, profitable trading, or reliable unattended replacement of original posts. Historical balanced event assessments are small and judgment based. General-English training text and public crypto posts can contain incomplete, biased, inaccurate, or disputed content; reconstruction does not verify it.

Financial quantities, units, inequalities, actor roles, timing windows, action status, and negation need particular care. Multiple reconstructions can share the same wrong detail. A fluent or recognizable account does not certify that those roles survived.

## License and attribution

Aparecium code and the learned nonlinear projection are licensed under MIT. The included original MPNet vocabulary weights and tokenizer retain their Apache-2.0 terms and applicable notices. The artifact includes upstream license material. Training-source terms remain with their owners, and public availability of an X post does not imply an open-content license.

Developed and maintained by [SentiChain](https://github.com/SentiChain). Source code, [usage documentation](https://github.com/SentiChain/aparecium/blob/master/README.md), and [issue tracking](https://github.com/SentiChain/aparecium/issues) are available in the [Aparecium repository](https://github.com/SentiChain/aparecium).
