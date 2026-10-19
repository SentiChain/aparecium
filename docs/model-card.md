---
language:
- en
license: mit
library_name: aparecium
pipeline_tag: text-generation
base_model: openai-community/gpt2
base_model_relation: adapter
tags:
- embedding-inversion
- sentence-embeddings
- lora
datasets:
- allenai/c4
- common-pile/news_filtered
- heegyu/news-category-dataset
- izumi-lab/wikinews-en-20230728
- sentence-transformers/wikipedia-en-sentences
- HuggingFaceFW/fineweb-edu
- webis/tldr-17
---

# Aparecium

Aparecium reconstructs short English text from a standard 768-dimensional [`all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) sentence embedding. A learned mapper converts the vector into a prefix for GPT-2 with LoRA adapters. Generated candidates are scored by their MPNet similarity to the input, and one is selected as the reconstruction.

The model provides an approximate interpretation of an embedding. It cannot guarantee recovery of the original wording or facts. Outputs can omit or invent names, quantities, dates, actions, and relationships.

## Loading

```bash
pip install aparecium
```

```python
from aparecium import Aparecium

model = Aparecium()
result = model.invert_text("Heavy rain forced the city to close several roads.")
print(result.text)
```

`invert_text` is a round-trip convenience method. To reconstruct existing embeddings, use `invert_embedding` or `invert_embeddings`; these methods do not need the original text. The input must come from the stock `sentence-transformers/all-mpnet-base-v2` encoder. Other encoders are not interchangeable.

This checkpoint uses the **Aparecium Python loader**. It is not a standalone Transformers `AutoModel` checkpoint or a generic text-prompt model. Loading requires the prefix mapper, LoRA adapter, pretrained GPT-2 base, and MPNet encoder. GPT-2 and MPNet are fetched separately and cached by their libraries.

The model repository contains `config.json`, `mapper.safetensors`, and the adapter files `adapter/adapter_config.json` and `adapter/adapter_model.safetensors`, alongside documentation and license notices. It does not contain optimizer state or require loading Python pickle checkpoints.

## Architecture

| Component | Configuration |
| --- | --- |
| Input encoder | Frozen `sentence-transformers/all-mpnet-base-v2`, 768 dimensions |
| Prefix mapper | Two linear layers with a GELU activation; hidden width 1,024 |
| Prefix length | 10 GPT-2 input positions |
| Language model | `openai-community/gpt2` |
| Adaptation | LoRA, rank 16, alpha 32, dropout 0.05 |
| Default candidate count | 5: one greedy candidate and four sampled candidates |
| Default generation limit | 64 tokens |
| Candidate selection | MPNet cosine similarity with a length and repetition filter |

The base language model is adapted through LoRA while the prefix mapper is trained directly. The MPNet encoder remains unchanged. The original text is used as a training target, but is not an input to reconstruction at inference time. The checkpoint configuration pins upstream GPT-2 and MPNet revisions.

## Training data and procedure

The corpus contains 300,000 short English texts drawn from news, encyclopedic and educational text, and social posts. The same 1,000 validation examples used for the earlier checkpoint were reserved, leaving **299,000 training examples**.

| Source | Corpus examples |
| --- | ---: |
| C4, realnewslike | 60,000 |
| Common Pile News, filtered to CC-BY documents | 30,000 |
| HuffPost News Category | 45,000 |
| Wikinews | 15,000 |
| Wikipedia sentences | 45,000 |
| FineWeb-Edu | 45,000 |
| Webis TL;DR | 60,000 |
| **Total, including validation** | **300,000** |

Texts were extracted as headlines, short summaries, or sentences, with a maximum of 60 GPT-2 tokens. Data preparation filtered duplicates and known semantic overlap with the evaluation sets. Dataset text is not included with this checkpoint.

Training used three passes, batches of eight, a mapper learning rate of 0.001, a LoRA learning rate of 0.0005, and a 5% warmup fraction. The final checkpoint completed 112,125 optimizer steps. The training objective was next-token prediction of the text conditioned on its embedding prefix, including an end-of-text token.

The training run took approximately 641 minutes on an Apple M5 GPU with a 4 GB memory cap. These are measurements of one run, not hardware requirements or cross-platform performance guarantees.

## Evaluation

The current checkpoint and an earlier checkpoint using the same architecture were compared on 1,222 held-out examples, using five candidates per input and matched generation settings. Candidate selection used MPNet; evaluation used the separate `BAAI/bge-small-en-v1.5` encoder.

| Measure | Earlier, 99,087-example corpus | Current, 300,000-example corpus |
| --- | ---: | ---: |
| Independent-similarity threshold passes | 132 / 1,222 (10.8%) | 223 / 1,222 (18.2%) |
| Average CPU seconds per item, batch size 8 | 0.44 | 0.41 |

The independent-similarity threshold is cosine similarity of 0.864, calibrated against STS-B labels. **These percentages are not factual accuracy rates.** With twenty candidates, the current checkpoint passed the same threshold for 287 / 1,222 examples (23.5%), averaging 1.65 CPU seconds per item in batches of eight.

This comparison uses one training seed and more total updates for the larger corpus, so it does not isolate the effect of data quantity. Improvements in similarity do not establish consistent improvement in every reconstructed claim. The comparison is with an earlier checkpoint of this architecture, not directly with the previous package decoder.

Recorded examples from a separate ten-sentence illustration, using five candidates and seed 42, show both useful recovery and remaining errors:

| Original | Reconstruction |
| --- | --- |
| Heavy rain forced the city to close several roads. | The city closed several roads due to heavy rain. |
| Scientists developed a battery that can be recharged in ten minutes. | Scientists have developed a rechargeable battery for ten seconds. |
| The company denied reports that it would lay off 500 employees. | The company denied hiring 500 employees. |

Full methods, group results, examples, and timing limitations are documented in the [benchmark report](https://github.com/SentiChain/aparecium/blob/master/docs/benchmarks.md).

## Intended use and limitations

The model supports inspecting the approximate meaning represented by short English MPNet embeddings and studying embedding reconstruction. It is not designed for exact quotation recovery, long-document reconstruction, other embedding models, or verified reporting.

Inspect critical details against an independent source before relying on a reconstruction. In particular, a fluent output can reverse an action or introduce an unsupported amount. The returned candidate scores measure embedding similarity, not confidence that the output is true. The model also inherits limitations and biases of its language model and training sources.

## License and attribution

Aparecium's code and released mapper/adapter weights are licensed under MIT. GPT-2 is an upstream MIT-licensed model; MPNet is distributed under Apache-2.0. Training datasets retain their own licenses and are not redistributed with the model. See [source acknowledgments](https://github.com/SentiChain/aparecium/blob/master/ATTRIBUTIONS.md).

Developed and maintained by [SentiChain](https://github.com/SentiChain). Source code and issue tracking are available in the [Aparecium repository](https://github.com/SentiChain/aparecium).
