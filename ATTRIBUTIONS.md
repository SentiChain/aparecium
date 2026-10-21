# Attributions

Aparecium's code and learned nonlinear projection are distributed under the [MIT License](LICENSE). The model artifact also contains the original MPNet input-vocabulary weights and tokenizer files, which retain their Apache-2.0 terms and applicable notices. The MIT license does not relicense those upstream components or source datasets.

The upstream terms are included in [UPSTREAM_LICENSE](UPSTREAM_LICENSE), with component notices in [NOTICE](NOTICE).

## Encoder and vocabulary

| Component | Source | Upstream license |
| --- | --- | --- |
| Input encoder, tokenizer, and original vocabulary weights | [all-mpnet-base-v2, Sentence Transformers](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | Apache-2.0 |

The supported encoder and tokenizer revision is `e8c3b32edf5434bc2275fc9bab85f82640a19130`. Matrix reconstruction loads the included tokenizer and vocabulary. The full encoder is obtained separately when producing matrices or using text convenience methods.

The implementation uses [PyTorch](https://github.com/pytorch/pytorch), [Transformers](https://github.com/huggingface/transformers), [Hugging Face Hub](https://github.com/huggingface/huggingface_hub), [Safetensors](https://github.com/huggingface/safetensors), and [NumPy](https://github.com/numpy/numpy). The optional service uses [FastAPI](https://github.com/fastapi/fastapi), [Pydantic](https://github.com/pydantic/pydantic), and [Uvicorn](https://github.com/encode/uvicorn). Their copyright and license notices remain with their distributions.

## Training sources

The general-English training collection contains 5,000 short texts from the sources below. Dataset-level license labels do not replace the terms or attribution attached to individual source documents.

| Collection | Training texts | Source and acknowledgment | Recorded license information |
| --- | ---: | --- | --- |
| C4, realnewslike | 978 | [Allen Institute for AI C4 distribution](https://huggingface.co/datasets/allenai/c4), derived from Common Crawl | ODC-By |
| Common Pile News | 499 | [Common Pile contributors](https://huggingface.co/datasets/common-pile/news_filtered) and original news authors | Document-level CC-BY filtering; retained documents record Attribution 4.0 |
| HuffPost News Category | 767 | [News Category Dataset by Rishabh Misra](https://rishabhmisra.github.io/publications/), via [heegyu/news-category-dataset](https://huggingface.co/datasets/heegyu/news-category-dataset) | CC-BY-4.0, as labeled by the collection |
| Wikinews | 270 | [Wikinews contributors](https://en.wikinews.org/), via [Izumi Lab's English collection](https://huggingface.co/datasets/izumi-lab/wikinews-en-20230728) | CC-BY-2.5 |
| Wikipedia sentences | 756 | [Wikipedia contributors](https://en.wikipedia.org/), via [Sentence Transformers' English collection](https://huggingface.co/datasets/sentence-transformers/wikipedia-en-sentences) | Wikipedia CC-BY-SA terms; the collection does not specify a version |
| FineWeb-Edu | 728 | [Hugging Face FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), derived from Common Crawl | ODC-By |
| Webis TL;DR | 1,002 | [Webis Group TL;DR collection](https://huggingface.co/datasets/webis/tldr-17) and original contributors | CC-BY-4.0 |
| **General-English total** | **5,000** | | |

Crypto training additionally uses 140 original public X posts across 133 event families. Those authors retain rights in their posts; public availability does not imply an open-content license. The source collections supply headlines, short summaries, sentences, and social commentary. Separate held-out diagnostic collections contain 500 general-English texts and 50 crypto posts.

The release distributes model parameters and tokenizer assets, not training or evaluation text. Original authors, corpus maintainers, and upstream notices remain the authorities for source content and its terms.

## Evaluation sources

Source-fidelity measurements use held-out public crypto posts, grouped by historical event and distinct author. The reported scores assess preservation of supplied source claims, not their external truth. Public documentation includes aggregate measurements and short paraphrased error descriptions without redistributing the underlying posts.
