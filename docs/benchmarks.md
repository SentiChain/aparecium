# Reconstruction benchmarks

Aparecium can retain the central meaning of short crypto posts while corrupting a consequential quantity, forecast horizon, or condition. The measurements below distinguish token recovery from preservation of source claims. They do not measure the truth of those claims, future returns, or trading profitability.

## Input and assessment

The reported model consumes complete native float32 token-state matrices from `sentence-transformers/all-mpnet-base-v2`, with encoder and tokenizer revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`. It receives no source text, reference token IDs, retrieved material, or source-derived constraints during reconstruction. Each visible interior state produces one vocabulary prediction. The first and last framing positions are removed before text decoding.

The principal collection contains 60 held-out original posts from 20 historical event families, with three distinct authors per family. Events cover February through June 2026 and are balanced as six bullish, six bearish, four mixed, and four unclear source contexts. This is a deliberately balanced historical collection, not a prospective test or a sample of live-news prevalence. Training and previously assessed event families were excluded from this collection.

Event accounts were formed from each family's three reconstructed posts, then assessed against the original source claims. Two source-fidelity assessments applied the same criteria, with a second interpretation of ambiguous details. These are judgment-based ratings and have not been independently human validated. The two assessments are sensitivity checks, not independent samples from a population.

A useful central claim means that the account conveys the event's main source-supported content. A critical assertion materially changes an asset, quantity, timing, action, condition, or other consequential claim. An account counted as useful without a critical assertion can still omit secondary details. These labels concern source fidelity and do not certify suitability for trading.

## Event-level results

Both assessments recorded the same principal event counts:

| Measure | Reconstructed-post accounts | Original-source accounts |
| --- | ---: | ---: |
| Useful central claim | 20 / 20 | 20 / 20 |
| Useful without a critical assertion | 18 / 20 | 20 / 20 |
| At least one critical assertion | 2 / 20 | 0 / 20 |
| Actionable event context | 20 / 20 | 20 / 20 |
| Material action status preserved | 20 / 20 | 20 / 20 |

Actionable context can coexist with a critical assertion. Recognizing the event and asset does not make a wrong amount or date acceptable. Original-source accounts are a control for what the supplied posts can support; their ratings do not establish the external truth of the posts.

Across separately composed historical collections of 20 families, useful accounts without critical assertions ranged from 13 to 18 out of 20, or 65% through 90%. Those collections contain different events. The variation cannot be interpreted as a performance trend, and the latest percentage is not a production accuracy estimate.

## Financial and timing details

Financial matching binds a value to its unit, asset, and economic role. Timing matching binds its value and kind to an actor, action, and timing role. Counts describe matched asserted fields (TP), unmatched asserted fields (FP), and missing source fields (FN). F1 is `2TP / (2TP + FP + FN)`.

The collection contains 48 central financial roles across 16 applicable events and 57 central timing roles across 20 events.

| Field measure | Primary TP / FP / FN | Primary F1 | Sensitivity TP / FP / FN | Sensitivity F1 |
| --- | ---: | ---: | ---: | ---: |
| Financial value, exact | 36 / 2 / 12 | 0.837 | 35 / 3 / 13 | 0.814 |
| Financial value, tolerated approximation | 37 / 1 / 11 | 0.860 | 36 / 2 / 12 | 0.837 |
| Timing role | 44 / 3 / 13 | 0.846 | 43 / 3 / 14 | 0.835 |

Tolerated matching permits contextual amounts within 2.5% and quoted prices within 0.5% only when recognizable rounding or approximation is appropriate. Exact-sensitive roles remain exact; timing has no numerical tolerance. These tolerances do not validate a lost inequality, condition, action status, or economic meaning.

Complete central financial coverage occurred in 9 of 16 events under the primary assessment and 8 of 16 under sensitivity assessment. Complete timing coverage occurred in 12 of 20 and 11 of 20 respectively. A high field-level F1 can therefore coexist with an event account that lacks an important role.

## Token recovery

Across the same 60 posts, the model recovered 1,439 of 1,713 body token slots excluding URLs, or 84.00%. This measures token identity in aligned source positions. It does not establish that an entire amount, sentence, or event claim was preserved. URLs and tokenization fragments can account for substantial additional output, and malformed fragments may be difficult to read.

## Consequential errors

These short descriptions paraphrase recorded cases rather than reproducing source posts:

| Source content | Reconstructed account | Practical consequence |
| --- | --- | --- |
| An exchange mistakenly credited about 2,000 BTC. | The account asserted about 2,000,000 BTC. | A 1,000-fold magnitude error survived an otherwise recognizable event account. |
| A long-term SOL price forecast used a 2030 horizon. | The account asserted a disagreement between 2030 and 2036. | The apparent source disagreement was introduced by reconstruction. |
| A collateral account distinguished $325 million of holdings from a $315 million strategy allocation. | The account listed both figures without preserving their distinct roles, and omitted a corrupted BTC quantity. | Readable context survived while financial coverage and economic interpretation were incomplete. |

Broad accumulation, product-launch, selloff, and policy context can remain readable and useful. Financial roles, bounds, actor relationships, and time scope require closer inspection. Three posts do not guarantee correction: correlated mistakes can reinforce one another when an account is assembled from multiple reconstructions.

## Runtime measurement

One recorded four-thread CPU run processed the 60 stored matrices in approximately 2.43 seconds. This timing covered shared prediction and associated validation work in the measurement path. It is not isolated decoder throughput or a single-request service latency guarantee. It excludes first-time downloads and does not establish cross-machine or sustained-load performance.

The matrices contained 2,714 visible state rows and 8,337,408 bytes of native float32 state data. Full token states require substantially more storage than a single pooled vector. Compression and production storage economics were not measured.

## Interpretation limits

- The event collection is small and deliberately balanced. Historical event selection and shared market episodes limit generalization to live traffic.
- Source-fidelity ratings are judgment based and have not been independently human validated. Financial matching and materiality decisions can differ even when event-level counts agree.
- The model was trained once with a fixed data mixture and update budget. These results do not measure variation across training seeds.
- Native state and framing checks establish compatibility with the specified encoder, not factual correctness. Dimension checks alone cannot verify provenance.
- Missing information and a false assertion have different consequences. Token recovery, field F1, and event usefulness measure different aspects of reconstruction.
- The measurements do not establish exact quotation recovery, long-document quality, other-language quality, profitability, or reliable unattended replacement of source posts.

See the [model card](https://github.com/SentiChain/aparecium/blob/master/docs/model-card.md) for architecture and training provenance, and the [README](https://github.com/SentiChain/aparecium/blob/master/README.md) for the input contract.
