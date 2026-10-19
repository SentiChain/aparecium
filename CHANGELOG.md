# Changelog

## 2.0.0 (2026-10-19)

- Replace the previous decoder with a learned prefix and GPT-2 LoRA adapters for reconstructing short English text from standard MPNet embeddings.
- Use the checkpoint trained with a 300,000-example corpus, including 1,000 examples reserved for validation. The default model is `SentiChain/aparecium`.
- Add batch methods, candidate similarity scores, selected-candidate indices, and end-of-text status to the public API.
- Provide a command-line interface and an optional HTTP service for the same model and generation settings.
- Increase the independent-similarity threshold pass count from 132 to 223 out of 1,222 held-out texts compared with the earlier 99,087-example checkpoint using the same architecture and five candidates. Similarity does not establish factual correctness.
- Replace the previous checkpoint format and remove surrogate scoring, beam search, and their configuration options. See the [migration guide](README.md#migrating-from-the-previous-release) before upgrading.

## 1.0.0

- Release the pooled MPNet reconstruction API and the earlier decoder for crypto-related text.
