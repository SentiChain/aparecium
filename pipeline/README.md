Pipeline: Training and Retraining with Cached Embeddings

This sample pipeline shows how to:
- Build a reusable SQLite database of post texts and their token-level MPNet embedding matrices (cached to avoid recomputation).
- Train a `Seq2SeqReverser` model from cached embeddings.
- Resume/retrain from a saved checkpoint.

Prerequisites
- Place your source text file at `data/crypto_posts_500k.txt` (one post per line), or pass `--input` to the scripts.
- Install the package and deps (from repo root):
  - `pip install -e .`

Configuration
- Defaults live in `pipeline/config.json`. You can override via CLI flags.

1) Prepare the database (cache 10k embeddings)

This step reads up to 10,000 posts, computes token-level MPNet embeddings, and stores both the raw text and embedding matrices into an SQLite DB for reuse.

Example:
```bash
python pipeline/prepare_db.py \
  --input data/crypto_posts_500k.txt \
  --db data/pipeline/aparecium_posts.db \
  --samples 10000 \
  --chunk-size 1000 \
  --model-name sentence-transformers/all-mpnet-base-v2 \
  --device auto
```

Notes:
- Set `--random-sample` to sample a uniform 10k subset deterministically (`--seed` controls reproducibility). Otherwise the first N lines are used.
- Embedding matrices are stored chunked by `block_start == block_end == chunk_id` to enable streaming retrieval during training.

2) Train or resume the model

Train a `Seq2SeqReverser` with batches built directly from the cached embeddings.

Example (fresh training):
```bash
python pipeline/train.py \
  --db data/pipeline/aparecium_posts.db \
  --model-dir models/pipeline_reverser \
  --epochs 3 \
  --batch-size 16 \
  --lr 1e-4 \
  --device auto
```

Resume training from an existing checkpoint:
```bash
python pipeline/train.py \
  --db data/pipeline/aparecium_posts.db \
  --model-dir models/pipeline_reverser \
  --epochs 2 \
  --batch-size 16 \
  --resume \
  --device auto
```

Implementation details
- Embedding cache: uses `aparecium.db_utils.ApareciumDB` and stores each chunk of posts and embeddings with unique `(block_start, block_end)` per chunk. Matrices are persisted as NumPy arrays to minimize size.
- Vectorization: uses `aparecium.Vectorizer` with MPNet (`sentence-transformers/all-mpnet-base-v2`).
- Training: uses `aparecium.Seq2SeqReverser.train_step_batch(...)` in mini-batches; checkpoints saved at `--model-dir` each epoch (and optionally mid-epoch via `--save-every`).

Tips
- If running on CPU, consider reducing `--chunk-size` and `--batch-size`.
- If you change the embedding model (`--model-name`), you must retrain since embeddings and tokenizer must align.

Storage footprint (rough estimate)
- MPNet hidden size is 768. For an average of ~64 tokens/post and float32 storage, each matrix is ~64×768×4 bytes ≈ 196 KB. For 10k posts this is ~1.9 GB, plus SQLite overhead.
- To reduce size, you can switch matrices to float16 by modifying `prepare_db.py` to cast arrays to `np.float16` before storage.


