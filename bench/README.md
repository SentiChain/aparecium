# Aparecium benchmark harness

Repository-only tooling (not shipped in the PyPI package) for measuring how well an
inverter recovers the *meaning* of a text from its stock `all-mpnet-base-v2` vector.
Test sets are frozen first; every model change is then scored on the same items.

Dataset text is written to `data/bench/` (gitignored) and is never committed.
Only ids, hashes, counts, and scores live in `bench/manifests/` and `bench/results/`.

## Run

```bash
python -m bench.build_testsets                    # frozen test sets A-D + calibration pairs E (+ vectors)
python -m bench.calibrate                         # what cosine counts as "same meaning"
python -m bench.build_trainset --size 10000 --name stage1
python -m bench.predict --system s1               # released v2 S1 checkpoint
python -m bench.predict --system nn --train stage1  # nearest-neighbour baseline
python -m bench.score --systems s1 nn_stage1 --report stage0
```

## Test sets (never trained on)

| Set | Source | What it measures |
|---|---|---|
| A | 50 held-out items from each training source | in-distribution ceiling |
| B | CNN/DailyMail highlights (first bullet), XSum one-sentence summaries | news from outlets absent from training |
| C | BBC article ledes (news/sport pages only), one full day per month Feb-Jun 2025 | new events/names; per-day "what is today about" |
| D | tweets (TweetEval), Stack Exchange titles, CoinDesk-era crypto headlines | other registers |
| E | STS-B test, PAWS test | calibration of cosine vs human "same meaning" |

Training draws only from openly licensed sources (see `bench/sources.py`):
C4 realnewslike (ODC-By), Common Pile news filtered to CC-BY, HuffPost News Category
(CC-BY-4.0), Wikinews (CC-BY-2.5), Wikipedia sentences (CC-BY-SA), FineWeb-Edu (ODC-By),
Webis TL;DR-17 (CC-BY-4.0). Test-only sources may carry research-only or unclear terms,
which is why their text is never redistributed.

A training candidate is dropped if it matches any test item by normalized-text hash,
or if it is "same meaning" as any test item by the calibrated bar on either encoder
(MPNet or the independent one), so same-event rewrites cannot leak across the split.

Reproducibility: sampling reads seeded random Parquet row groups (`bench/fetch.py`),
each source pool is pinned to the Hub Parquet-export commit it was first built from,
all embeddings are computed on CPU, and caches are keyed by a digest of the exact
texts. `bench/manifests/` records pool commits, encoder revisions and library
versions, and SHA-256 checksums of every frozen vector file.

## Metrics

- **Same meaning (MPNet / independent)**: share of reconstructions whose cosine to the
  original reaches the level STS-B humans rate >= 4/5, measured with MPNet and with an
  unrelated encoder (`BAAI/bge-small-en-v1.5`) so MPNet-specific gaming is visible.
- **Cos MPNet / Cos indep.**: mean cosine between reconstruction and original.
- **Token F1, BLEU, ROUGE-L, exact**: wording overlap (secondary; the goal is meaning).
- **Names/numbers kept**: share of the original's numbers, tickers, and capitalized
  non-initial words that appear in the reconstruction.
- **Stops by itself / distinct candidates**: decoder health checks.
- **Day gist (set C)**: cosine between a day's average original vector and average
  reconstruction vector.
