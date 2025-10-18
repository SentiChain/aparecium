
# Aparecium‑V2 (Pooled‑Only)

**Original, pooled‑vector‑only embedding inversion for crypto tweets.**  
This repo accepts a single 768‑D `all-mpnet-base-v2` pooled embedding and reconstructs text using:
- Multi‑channel EmbAdapter (pooled → pseudo‑sequence)
- Sketcher (content plan) + constrained decoding (light penalty now; trie/FSM planned)
- Surrogate similarity scorer `r(x,e)` for sequence‑level reranking
- Optional RL fine‑tuning (SCST)

## Quickstart

1) Embed raw tweets into pooled MPNet vectors:
```bash
python -m aparecium_v2.scripts.embed_mpnet --input raw_tweets.jsonl --output embeddings.jsonl
```

2) Supervised train (S1):
```bash
export TRAIN_JSONL_DIR=path/to/train_dir_or_file
export VAL_JSONL_DIR=path/to/val_dir_or_file
python -m aparecium_v2.train.train_s1_supervised --shards $TRAIN_JSONL_DIR --save_dir checkpoints
```

3) Train surrogate scorer `r`:
```bash
export TRAIN_JSONL_DIR=path/to/train_dir_or_file
export VAL_JSONL_DIR=path/to/val_dir_or_file
python -m aparecium_v2.train.train_surrogate_r
```

4) RL fine‑tuning (S2, SCST):
```bash
export S1_CHECKPOINT=checkpoints/aparecium_v2_s1.pt
python -m aparecium_v2.train.train_s2_scst --shards $TRAIN_JSONL_DIR --ckpt_s1 $S1_CHECKPOINT --save_dir checkpoints
```

5) Invert a single embedding (JSON from stdin):
```bash
echo '{"e":[0.0,0.1,...]}' | python -m aparecium_v2.scripts.invert_once --ckpt checkpoints/aparecium_v2_s1.pt
```

## Repo layout
See the `aparecium_v2/` package for modules:
- `models/` — EmbAdapter, Decoder, SurrogateR, Sketcher, Constraints
- `train/`  — S1, S2 (SCST), surrogate `r` trainer
- `infer/`  — Deterministic beam + surrogate reranking (+ light constraints)
- `data/`   — dataset/plan extractors
- `scripts/`— embedding, sharding, invert_once
- `eval/`   — simple metrics & eval harness

## Notes
- This design is **not** vec2text: no per‑step embedder calls; single‑pass decode with plan and surrogate rerank.
- Input contract is strictly **pooled 768‑D vector** (MPNet v2).

## Inference service
Run the FastAPI service:
```bash
python -m aparecium_v2.infer.service --ckpt checkpoints/aparecium_v2_s2.pt
```
POST `/invert` with:
```json
{ "embedding": [float;768], "deterministic": true, "beam": 5, "max_len": 64, "constraints": true }
```
Response includes `text`, `candidates`, `scores.lm_logp[]`, optional `plan`, and `version`.

## License
MIT (add license file).
