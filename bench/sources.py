"""Dataset sources for the benchmark: where each text unit comes from and how it is cut.

Every extractor turns one dataset row into at most one short text unit
(headline, lede, sentence, or post) that fits the decoder budget, so no single
article contributes more than one unit (avoids headline-in-test /
lede-in-train leakage).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .common import clean_text, fits, lede, split_sentences, stable_int, SEED


@dataclass
class Source:
    key: str
    repo: str
    config: Optional[str]
    split: str
    license: str
    group: str  # news / general / social / other / calibration
    extract: Callable[[Dict, str], Optional[Dict]] = field(repr=False)
    note: str = ""
    columns: List[str] = field(default_factory=list)


def _unit(uid: str, text: str) -> Optional[Dict]:
    text = clean_text(text)
    if not fits(text):
        return None
    return {"id": uid, "text": text}


def _first_long_paragraph(text: str, min_words: int = 12) -> str:
    for para in (text or "").split("\n"):
        if len(para.split()) >= min_words:
            return para
    return ""


_TERMINAL = re.compile(r"[.!?][\"'”’)\]]*$")
_LIST_MARKER = re.compile(r"^\s*(?:[-*•●▪]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])\s")
_CREDIT = re.compile(
    r"\b(photo|image|picture|screenshot|screencap|credit|courtesy|used with permission|"
    r"cc by|creative commons|via flickr|all rights reserved)\b", re.I)


def _is_prose(s: str) -> bool:
    """A complete sentence: ends with terminal punctuation, is not a list item."""
    return bool(_TERMINAL.search(s)) and not _LIST_MARKER.match(s)


# ---------------------------------------------------------------- training sources


_C4_PREFIX = re.compile(r"^(?:[^|.!?]{1,40}\|\s*)+|^[A-Z][A-Za-z .]{0,24}:\s+")


def ex_c4(row, rid):
    para = _first_long_paragraph(row.get("text", ""))
    if not para:
        return None
    return _unit(f"c4:{row.get('url') or rid}", lede(_C4_PREFIX.sub("", clean_text(para))))


def _commonpile_license(meta) -> str:
    if isinstance(meta, str):
        try:
            meta = ast.literal_eval(meta)
        except Exception:
            return meta
    return (meta or {}).get("license", "") if isinstance(meta, dict) else ""


def ex_commonpile(row, rid):
    lic = _commonpile_license(row.get("metadata"))
    # Keep plain CC-BY only (drop ShareAlike / NonCommercial / NoDerivatives).
    if "Attribution" not in lic or any(
        x in lic for x in ("Share", "NonCommercial", "NoDeriv")
    ):
        return None
    lines = [x.strip() for x in (row.get("text") or "").split("\n") if x.strip()]
    if len(lines) < 4:
        return None
    uid = f"cpnews:{row.get('source')}:{row.get('id')}"
    if stable_int(uid, SEED) % 2 == 0:
        title = re.split(r"\s+[-·|–—]\s+(?=[^-·|–—]*$)", lines[0])[0]
        return _unit(uid, title)
    body = [x for x in lines[1:]
            if not x.startswith("Published") and len(x.split()) >= 12 and not _CREDIT.search(x)]
    return _unit(uid, lede(clean_text(body[0]))) if body else None


def ex_huffpost(row, rid):
    uid = f"huffpost:{row.get('link') or rid}"
    if stable_int(uid, SEED) % 2 == 0:
        return _unit(uid, row.get("headline", ""))
    return _unit(uid, row.get("short_description", ""))


def ex_wikinews(row, rid):
    uid = f"wikinews:{row.get('url') or rid}"
    if stable_int(uid, SEED) % 2 == 0:
        return _unit(uid, row.get("title", ""))
    para = _first_long_paragraph(row.get("text", ""))
    return _unit(uid, lede(clean_text(para))) if para else None


def ex_wiki_sentence(row, rid):
    s = clean_text(row.get("sentence", ""))
    return _unit(f"wikisent:{rid}", s) if _is_prose(s) else None


def ex_fineweb(row, rid):
    paras = [p for p in (row.get("text") or "").split("\n") if len(p.split()) >= 12]
    if not paras:
        return None
    uid = f"fineweb:{row.get('id') or rid}"
    para = paras[stable_int(uid, SEED) % len(paras)]
    sents = [s for s in split_sentences(clean_text(para)) if len(s.split()) >= 6 and _is_prose(s)]
    if not sents:
        return None
    return _unit(uid, sents[stable_int(uid, SEED, "s") % len(sents)])


def ex_tldr(row, rid):
    return _unit(f"tldr:{row.get('id') or rid}", row.get("summary", ""))


TRAIN_SOURCES = {
    s.key: s
    for s in [
        Source("c4_news", "allenai/c4", "realnewslike", "train", "ODC-By", "news", ex_c4,
               "first 1-3 sentences of a news article (2019 crawl)", ["text", "url"]),
        Source("commonpile_news", "common-pile/news_filtered", None, "train",
               "CC-BY 4.0 (filtered per document)", "news", ex_commonpile,
               "title or lede of CC-BY news articles", ["id", "text", "source", "metadata"]),
        Source("huffpost", "heegyu/news-category-dataset", None, "train", "CC-BY-4.0",
               "news", ex_huffpost, "HuffPost headline or one-line summary, 2012-2022",
               ["link", "headline", "short_description"]),
        Source("wikinews", "izumi-lab/wikinews-en-20230728", None, "train", "CC-BY-2.5",
               "news", ex_wikinews, "Wikinews title or lede", ["title", "text", "url"]),
        Source("wiki_sentences", "sentence-transformers/wikipedia-en-sentences", None,
               "train", "CC-BY-SA (Wikipedia)", "general", ex_wiki_sentence,
               "single Wikipedia sentence", ["sentence"]),
        Source("fineweb_edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train",
               "ODC-By", "general", ex_fineweb, "one sentence from an educational web page",
               ["id", "text"]),
        Source("tldr", "webis/tldr-17", None, "train", "CC-BY-4.0", "social", ex_tldr,
               "Reddit author-written TL;DR", ["id", "summary"]),
    ]
}

# Stage-1 quotas (10k total): ~50% news, ~30% general, ~20% social.
STAGE_QUOTA_WEIGHTS = {
    "c4_news": 0.20,
    "commonpile_news": 0.10,
    "huffpost": 0.15,
    "wikinews": 0.05,
    "wiki_sentences": 0.15,
    "fineweb_edu": 0.15,
    "tldr": 0.20,
}


# ---------------------------------------------------------------- test-only sources


def ex_cnndm(row, rid):
    first = (row.get("highlights") or "").split("\n")[0]
    first = re.sub(r"\s+([.,;:!?'])", r"\1", first)  # undo " ." tokenization spacing
    return _unit(f"cnndm:{row.get('id') or rid}", first)


def ex_xsum(row, rid):
    return _unit(f"xsum:{row.get('id') or rid}", row.get("summary", ""))


_BBC_ARTICLE = re.compile(r"bbc\.(?:co\.uk|com)/(?:news|sport)/(?!.*(?:/live/|newsletter))")


def ex_bbc(row, rid):
    if not _BBC_ARTICLE.search(row.get("link") or ""):
        return None  # skip live pages, audio, newsletters and promo blurbs
    u = _unit(f"bbc:{row.get('link') or rid}", row.get("description", ""))
    if u:
        u["day"] = str(row.get("published_date", ""))[:10]
    return u


def ex_tweet(row, rid):
    return _unit(f"tweet_eval:sentiment:test:{rid}", row.get("text", ""))


def ex_se_title(row, rid):
    return _unit(f"stackexchange:{rid}", row.get("title1", ""))


def ex_crypto(row, rid):
    return _unit(f"coindesk:{row.get('id') or rid}", row.get("title", ""))


TEST_SOURCES = {
    s.key: s
    for s in [
        Source("cnn_dm", "abisee/cnn_dailymail", "3.0.0", "test",
               "apache-2.0 card (CNN / Daily Mail copyright)", "news", ex_cnndm,
               "first highlight bullet", ["id", "highlights"]),
        Source("xsum", "EdinburghNLP/xsum", None, "test", "unknown (BBC copyright)",
               "news", ex_xsum, "one-sentence BBC summary", ["id", "summary"]),
        Source("bbc_2025", "RealTimeData/bbc_news_alltime", None, "train",
               "not stated (BBC copyright)", "news", ex_bbc,
               "BBC one-sentence lede, Feb-Jun 2025, grouped by day",
               ["link", "description", "published_date"]),
        Source("tweets", "cardiffnlp/tweet_eval", "sentiment", "test",
               "unknown (X ToS: no redistribution)", "other", ex_tweet, "tweet", ["text"]),
        Source("stackexchange", "sentence-transformers/stackexchange-duplicates",
               "title-title-pair", "train", "CC-BY-SA (Stack Exchange)", "other",
               ex_se_title, "question title", ["title1"]),
        Source("crypto", "maryamfakhari/crypto-news-coindesk-2020-2025", None, "train",
               "CC-BY-NC-4.0", "other", ex_crypto, "crypto news headline", ["id", "title"]),
    ]
}

BBC_MONTHS = ["2025-02", "2025-03", "2025-04", "2025-05", "2025-06"]


CALIBRATION_SOURCES = {
    s.key: s
    for s in [
        Source("stsb", "sentence-transformers/stsb", None, "test", "unknown", "calibration",
               lambda row, rid: {**row, "id": f"stsb:test:{rid}"}, "STS-B test pairs",
               ["sentence1", "sentence2", "score"]),
        Source("paws", "google-research-datasets/paws", "labeled_final", "test", "free use",
               "calibration", lambda row, rid: {**row, "id": f"paws:test:{row['id']}"},
               "PAWS labeled test pairs", ["id", "sentence1", "sentence2", "label"]),
    ]
}
