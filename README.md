# Aparecium

Aparecium reconstructs short English text from the full token-state matrix produced by [`sentence-transformers/all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). A nonlinear projection maps each token state to the encoder's original vocabulary. Reconstruction receives the matrix without requiring the source text.

**The result is an approximate reconstruction.** Names, numbers, dates, conditions, and actions can be omitted or changed. Use the output to inspect represented content, and check important claims against source material. It is not verified news or a basis for unattended trading decisions.

The input is `AutoModel.last_hidden_state`, including its first and last framing rows, from the pinned encoder and tokenizer described below. Pooled sentence vectors, including 768-element vectors from `SentenceTransformer.encode`, cannot be used. A stored pooled vector does not contain the full token states needed by this model.

## Install

```bash
pip install aparecium
```

The default model is [`SentiChain/aparecium-matrix`](https://huggingface.co/SentiChain/aparecium-matrix). Its weights and tokenizer are downloaded on first use and cached locally.

## Reconstruct a stored matrix

```python
import numpy as np
from aparecium import Aparecium

matrix = np.load("matrix.npy", allow_pickle=False)
model = Aparecium()
result = model.invert_matrix(matrix)

print(result.text)
print(result.token_ids)
print(result.input_rows)
```

`invert_matrix` uses only the supplied states and optional attention mask. It does not re-embed text or retrieve source material.

| Result field | Meaning |
| --- | --- |
| `text` | Decoded reconstruction, with special tokens skipped. |
| `token_ids` | Predicted IDs for every visible interior row, including any predicted special IDs. |
| `input_rows` | Number of visible input rows, including the first and last framing rows. |

The input contract is:

- Token states with shape `(sequence_length, 768)`, containing 3 through 256 rows. Preserve native `float32` encoder output when storing states.
- States from the pinned encoder and tokenizer, without pooling, row normalization, quantization, or dimensionality reduction.
- The first and last visible rows are the encoder's beginning and end framing positions. The decoder removes these positions before returning token IDs.
- An optional binary attention mask of the same sequence length. Visible rows must form a prefix followed by any right padding, with at least three visible rows.
- Finite real values only. Numeric inputs are converted to `float32`; overflow is rejected. Other embedding models are incompatible even if their output width is 768.

To reconstruct up to 32 matrices, pass a batch array or a list of matrices. Lists can contain different sequence lengths:

```python
results = model.invert_matrices([matrix, matrix])
for result in results:
    print(result.text)
```

Use `attention_masks=` when a batch contains padded rows. Matrix width and shape checks cannot establish encoder provenance: record the encoder identifier, revision, and preprocessing alongside stored states.

## Produce compatible matrices

Use the encoder and tokenizer at revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`. Save the complete final hidden states rather than a pooled or normalized sentence vector:

```python
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

encoder_id = "sentence-transformers/all-mpnet-base-v2"
encoder_revision = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
tokenizer = AutoTokenizer.from_pretrained(encoder_id, revision=encoder_revision)
encoder = AutoModel.from_pretrained(
    encoder_id, revision=encoder_revision, dtype=torch.float32
).eval()

inputs = tokenizer(
    "Heavy rain forced the city to close several roads.",
    return_tensors="pt",
    truncation=False,
)
if not 3 <= inputs["input_ids"].shape[1] <= 256:
    raise ValueError("Text must produce 3 through 256 token rows, including framing.")

with torch.inference_mode():
    matrix = encoder(**inputs).last_hidden_state[0].cpu().numpy()

np.save("matrix.npy", matrix, allow_pickle=False)
```

For a batch producer, retain each attention mask and keep only its visible prefix or pass the mask with its padded matrix. Do not truncate overlong source text silently: reconstruction can recover only the content represented by the stored states.

## Text convenience methods

`invert_text` creates compatible states from one string before reconstructing them. `invert_texts` accepts up to 32 strings. These are round-trip convenience methods; stored-matrix reconstruction uses `invert_matrix` or `invert_matrices`.

```python
result = model.invert_text("Heavy rain forced the city to close several roads.")
print(result.text)
```

These methods use the pinned `AutoTokenizer` and `AutoModel.last_hidden_state`, with no pooling. Text that produces more than 256 rows is rejected. The encoder weights are downloaded separately when a text convenience method first needs them.

## Loading and devices

```python
model = Aparecium(
    model_id="SentiChain/aparecium-matrix",
    revision=None,
    model_path=None,
    device=None,
    local_files_only=False,
    token_batch_size=256,
)
```

CPU is the default device. Set `device="mps"` explicitly to use a supported Apple GPU. The runtime supports CPU and MPS, and an unavailable requested device raises an error. `token_batch_size` accepts 1 through 256 and controls how many state rows are projected together; it does not alter the sequence-length or matrix-count limits.

Set `revision` to a Hub commit for reproducible model loading, or use `model_path` for a local artifact directory. `local_files_only=True` prevents downloads. Matrix reconstruction then needs the complete local model artifact; text convenience methods also need the pinned encoder files in the local cache.

## Command line

```bash
aparecium --matrix-file matrix.npy
aparecium --matrices-file matrices.json --json
aparecium "Heavy rain forced the city to close several roads."
echo "Heavy rain forced the city to close several roads." | aparecium --json
```

The equivalent module command is `python -m aparecium`. Choose a matrix file, a matrices file, or text. Without a file or positional text, the command reads text from standard input.

`--matrix-file` accepts a two-dimensional `.npy` file, a JSON matrix array, or a JSON object containing `matrix` and an optional `attention_mask`. `--matrices-file` accepts a three-dimensional `.npy` file, a JSON list of matrices, or a JSON object containing `matrices` and optional `attention_masks`. JSON supports matrices with different sequence lengths. Input files are limited to 16 MiB.

`--json` returns the result fields above. Batch JSON has a `results` array. Shared options are `--model-id`, `--revision`, `--model-path`, `--device`, `--local-files-only`, and `--token-batch-size`.

## HTTP service

```bash
pip install 'aparecium[server]'
aparecium-serve --host 127.0.0.1 --port 8000
```

The equivalent module command is `python -m aparecium.service`. The service accepts the shared model options above. `APARECIUM_LOCAL_FILES_ONLY=true` also enables offline loading.

`GET /health` reports readiness and the input contract. `POST /invert` accepts either `matrix` with optional `attention_mask`, or `matrices` with optional `attention_masks`. Supply one of those two forms and no other fields. The service does not need source text.

This example sends a complete saved matrix:

```python
import json
from urllib.request import Request, urlopen
import numpy as np

matrix = np.load("matrix.npy", allow_pickle=False)
request = Request(
    "http://127.0.0.1:8000/invert",
    data=json.dumps({"matrix": matrix.tolist()}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urlopen(request) as response:
    print(json.load(response))
```

A single request returns `text`, `token_ids`, and `input_rows`; a batch request returns a `results` array. Unknown fields, pooled vectors, invalid masks, and nonfinite values are rejected. Request bodies are limited to 16 MiB. Invalid inputs return HTTP 422, oversized bodies return 413, and an unavailable model returns 503.

## Quality and limitations

On a held-out collection of 20 historical crypto event families with three posts per event, all 20 reconstructed event accounts retained a useful central claim. Under two source-fidelity assessments, 18 were useful without a critical assertion and two contained critical assertions. These are judgment-based assessments, not independently human-validated accuracy estimates or verified market outcomes. The collection was balanced across bullish, bearish, mixed, and unclear source context rather than sampled from a live news stream.

Token recovery and broad event usefulness do not guarantee safe financial detail. In recorded cases, about 2,000 BTC became about 2,000,000 BTC, and a forecast horizon of 2030 became 2036. Forecast bounds, secondary quantities, and conditions can also disappear. Combining multiple reconstructed posts can repeat a shared error.

The model is intended for short English content from the specified encoder. Exact quotation recovery, long documents, other languages, trading profitability, and unattended source replacement have not been established. See the [benchmark details](https://github.com/SentiChain/aparecium/blob/master/docs/benchmarks.md) and [model card](https://github.com/SentiChain/aparecium/blob/master/docs/model-card.md) for training provenance, field-level results, and measurement limits.

## Input compatibility

Integrations that store pooled sentence vectors must capture full token states to use this model. Replace pooled-vector calls with `invert_matrix` or `invert_matrices`, preserve framing and attention masks, and update service payloads to `matrix` or `matrices`. Pooled checkpoints and vector APIs are unsupported.

## License and attribution

Aparecium code and the learned projection are licensed under the [MIT License](https://github.com/SentiChain/aparecium/blob/master/LICENSE). The included original MPNet vocabulary weights and tokenizer retain their Apache-2.0 terms and notices. Training sources retain their own terms; source text is not redistributed with the model. See [ATTRIBUTIONS.md](https://github.com/SentiChain/aparecium/blob/master/ATTRIBUTIONS.md).
