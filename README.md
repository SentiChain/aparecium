# Aparecium

[![PyPI](https://img.shields.io/pypi/v/aparecium)](https://pypi.org/project/aparecium/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/SentiChain/aparecium/blob/master/LICENSE)

Aparecium reconstructs short English text from a sentence embedding. It accepts the standard 768-dimensional vectors produced by [`sentence-transformers/all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2), then uses a learned prefix and GPT-2 with LoRA to generate candidate reconstructions. Candidates are checked for length and repeated phrases, then selected by MPNet similarity to the input.

**The result is an approximate reconstruction, not the original text.** Outputs can preserve a topic while omitting or inventing names, numbers, dates, or relationships. Similarity scores measure embedding alignment; they are not probabilities of factual correctness.

The current checkpoint uses a 300,000-example English corpus and replaces the previous decoder. The encoder remains unchanged: embeddings created with stock MPNet can be used directly. Vectors from other embedding models are not supported, even when they also have 768 dimensions.

## Install

```bash
pip install aparecium
```

Model files are downloaded on first use and cached locally. The default checkpoint is [`SentiChain/aparecium`](https://huggingface.co/SentiChain/aparecium). GPT-2 and MPNet are also downloaded as needed.

## Reconstruct an embedding

```python
from aparecium import Aparecium
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embedding = encoder.encode(
    "Heavy rain forced the city to close several roads.",
    normalize_embeddings=True,
)

model = Aparecium()
result = model.invert_embedding(embedding)

print(result.text)
print(result.candidates)
print(result.scores)
```

The reconstruction step receives only the embedding. The original sentence above is used only to create that embedding.

`InvertResult` exposes:

| Field | Meaning |
| --- | --- |
| `text` | Selected reconstruction. |
| `candidates` | Generated candidate texts. |
| `scores` | MPNet cosine similarity for each candidate, in the same order. |
| `selected_index` | Index of the selected candidate. |
| `stopped` | Whether the selected candidate emitted its end-of-text token. |

Use `invert_embeddings` to reconstruct multiple vectors:

```python
embeddings = encoder.encode(
    [
        "Heavy rain forced the city to close several roads.",
        "The airport reopened on Monday after a snow storm.",
    ],
    normalize_embeddings=True,
)

results = model.invert_embeddings(embeddings, num_candidates=5, max_length=64, seed=0)
for result in results:
    print(result.text)
```

For a quick round-trip check, `invert_text` embeds one string before reconstructing it; `invert_texts` does the same for a sequence of strings:

```python
result = model.invert_text("Heavy rain forced the city to close several roads.")
print(result.text)
```

## Generation settings

All four inversion methods accept `num_candidates`, `max_length`, and `seed`.

| Option | Default | Effect |
| --- | --- | --- |
| `num_candidates` | `5` | Accepts 1–32 candidates: one greedy candidate and additional sampled candidates. |
| `max_length` | `64` | Maximum generated tokens per candidate. |
| `seed` | `0` | Controls candidate sampling. |

Twenty candidates improved aggregate similarity in the benchmark, with about four times the inference time. More candidates do not guarantee a better reconstruction for an individual input. Sampling results can vary with hardware, dependency versions, and batch composition.

The default device is CUDA when available, otherwise CPU. Apple GPU execution requires an explicit `device="mps"`. Use an explicit device when needed:

```python
model = Aparecium(model_id="SentiChain/aparecium", device="cpu")
```

The constructor also accepts `revision` to select a specific Hub commit and `model_path` to load a local checkpoint directory. `local_files_only=True` prevents downloads; the checkpoint and the required GPT-2 and MPNet files must already be available locally.

## Command line

```bash
python -m aparecium "Heavy rain forced the city to close several roads."
```

```bash
echo "Heavy rain forced the city to close several roads." | python -m aparecium --json
```

The command embeds the supplied text and prints its reconstruction. Use `--json` to inspect the candidates and scores. Options include `--model-id`, `--revision`, `--model-path`, `--device`, `--candidates`, `--max-length`, and `--seed`.

## HTTP service

```bash
pip install 'aparecium[server]'
aparecium-serve
```

The equivalent module command is `python -m aparecium.service`. The service exposes `GET /health` and `POST /invert`. An inversion request contains either one `embedding` or multiple `embeddings`, plus optional `num_candidates`, `max_length`, and `seed` fields. It does not require the source text.

This example sends a complete vector:

```python
import json
from urllib.request import Request, urlopen
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embedding = encoder.encode(
    "Heavy rain forced the city to close several roads.",
    normalize_embeddings=True,
)
request = Request(
    "http://127.0.0.1:8000/invert",
    data=json.dumps({"embedding": embedding.tolist(), "num_candidates": 5}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urlopen(request) as response:
    print(json.load(response))
```

## Quality and limitations

On the same 1,222 held-out texts, using five candidates per input, the current model exceeded a calibrated independent-similarity threshold on **223 examples (18.2%)**, compared with **132 examples (10.8%)** for the earlier checkpoint trained with a 99,087-example corpus. This is a semantic-similarity proxy, not a factual accuracy rate or a direct comparison with the previous package decoder.

The following are recorded outputs from a separate ten-sentence illustration, using five candidates and seed 42:

| Original | Current checkpoint reconstruction |
| --- | --- |
| Heavy rain forced the city to close several roads. | The city closed several roads due to heavy rain. |
| Scientists developed a battery that can be recharged in ten minutes. | Scientists have developed a rechargeable battery for ten seconds. |
| The company denied reports that it would lay off 500 employees. | The company denied hiring 500 employees. |

The rain example preserves the claim. The battery example changes the duration, and the employment example changes the action being denied. Fluent output and a high similarity score do not establish that important facts survived.

The training corpus contains short English texts of up to 60 GPT-2 tokens. The model is not designed to recover long articles, exact quotations, or details absent from the pooled embedding. Treat reconstructions as hypotheses to inspect, not as verified source material.

See [benchmark details](https://github.com/SentiChain/aparecium/blob/master/docs/benchmarks.md) for the comparison, decoding costs, and measurement limitations, and the [model card](https://github.com/SentiChain/aparecium/blob/master/docs/model-card.md) for training and architecture information.

## Migrating from the previous release

The current release replaces the decoder and checkpoint format. Previous checkpoints cannot be loaded by the new implementation. `Aparecium()`, `invert_embedding`, `invert_text`, and the result's `text` and `candidates` fields remain the basic entry points.

| Previous option | Current option |
| --- | --- |
| `repo_id` | `model_id` |
| `ckpt_filename`, `r_ckpt_filename` | Use a model repository and `revision`, or a complete local `model_path`. |
| `beam` | `num_candidates`; this selects the number of generated candidates, not a beam width. |
| `max_len` | `max_length` |
| `rerank_every`, `alpha` | Removed with surrogate scoring and beam search. |
| Previous inference service module | `aparecium-serve` or `python -m aparecium.service`. |

Update these arguments when upgrading. The new implementation does not load the previous decoder.

## License and attribution

Aparecium code and released mapper/adapter weights are licensed under the [MIT License](https://github.com/SentiChain/aparecium/blob/master/LICENSE). GPT-2, MPNet, and the training datasets retain their respective licenses. See [ATTRIBUTIONS.md](https://github.com/SentiChain/aparecium/blob/master/ATTRIBUTIONS.md) for sources and acknowledgments.
