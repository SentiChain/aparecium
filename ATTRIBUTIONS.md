# Attributions

Aparecium's code and released prefix-mapper and LoRA-adapter weights are distributed under the [MIT License](LICENSE). Upstream models and training datasets retain their respective licenses. Aparecium does not relicense or redistribute the training corpus.

## Models

| Component | Source | Upstream license |
| --- | --- | --- |
| Pretrained language model | [GPT-2, OpenAI](https://huggingface.co/openai-community/gpt2) | MIT |
| Sentence encoder and candidate scorer | [all-mpnet-base-v2, Sentence Transformers](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | Apache-2.0 |
| Independent evaluation encoder | [bge-small-en-v1.5, BAAI](https://huggingface.co/BAAI/bge-small-en-v1.5) | MIT |

The GPT-2 and MPNet base weights are obtained separately from their upstream repositories. BGE is used for evaluation and is not required for inference.

The implementation uses [PyTorch](https://github.com/pytorch/pytorch), [Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), and [Sentence Transformers](https://github.com/huggingface/sentence-transformers). Their copyright and license notices remain with their distributions.

## Training sources

The table identifies the source collections and the license information recorded for them. Dataset-level license labels do not replace the terms or attribution attached to individual source documents.

| Collection | Source and acknowledgment | License information |
| --- | --- | --- |
| C4, realnewslike | [Allen Institute for AI C4 distribution](https://huggingface.co/datasets/allenai/c4), derived from Common Crawl | ODC-By |
| Common Pile News | [Common Pile contributors](https://huggingface.co/datasets/common-pile/news_filtered) and original news authors | Document-level CC-BY filtering; retained documents record Attribution 4.0 |
| HuffPost News Category | [News Category Dataset by Rishabh Misra](https://rishabhmisra.github.io/publications/), via [heegyu/news-category-dataset](https://huggingface.co/datasets/heegyu/news-category-dataset) | CC-BY-4.0, as labeled by the collection |
| Wikinews | [Wikinews contributors](https://en.wikinews.org/), via [Izumi Lab's English collection](https://huggingface.co/datasets/izumi-lab/wikinews-en-20230728) | CC-BY-2.5 |
| Wikipedia sentences | [Wikipedia contributors](https://en.wikipedia.org/), via [Sentence Transformers' English collection](https://huggingface.co/datasets/sentence-transformers/wikipedia-en-sentences) | Wikipedia CC-BY-SA terms; the collection does not specify a version |
| FineWeb-Edu | [Hugging Face FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), derived from Common Crawl | ODC-By |
| Webis TL;DR | [Webis Group TL;DR collection](https://huggingface.co/datasets/webis/tldr-17) and original contributors | CC-BY-4.0 |

Aparecium extracts short headlines, summaries, and sentences for training. The release contains model parameters, not copies of these datasets. Source authors, corpus maintainers, and their upstream notices remain the authorities for the original content and its terms.

## Evaluation sources

Evaluation includes held-out texts from the training sources, CNN/DailyMail and XSum summaries, BBC news descriptions, social posts, Q&A titles, and crypto headlines. The independent similarity threshold is calibrated using [STS Benchmark](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark). Aggregate scores and illustrative reconstructions do not change the licenses of those sources.
