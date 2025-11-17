import re
from typing import Dict, List, Any

# Simple regex/gazetteer-based plan extraction for crypto tweets.
TICKER_RE = re.compile(r"\$[A-Z]{2,10}")
HASHTAG_RE = re.compile(r"#[A-Za-z0-9_]+")
ETH_RE = re.compile(r"0x[a-fA-F0-9]{40}")
BTC_BECH32_RE = re.compile(r"bc1[ac-hj-np-z02-9]{11,71}")
SOL_BASE58_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")
NUM_RE = re.compile(r"(\$?\d{1,3}(,\d{3})*(\.\d+)?|\d+(\.\d+)?%)")
URL_RE = re.compile(r"https?://[^\s]+")


def extract_plan(text: str) -> Dict[str, Any]:
    tickers = list(set([t[1:] for t in TICKER_RE.findall(text)]))
    hashtags = list(set(HASHTAG_RE.findall(text)))
    # addresses (basic; you can extend with better validators)
    addrs = set(ETH_RE.findall(text))
    addrs |= set(x for x in text.split() if BTC_BECH32_RE.match(x))
    addrs |= set(x for x in text.split() if SOL_BASE58_RE.match(x))
    amounts = list(set(NUM_RE.findall(text)))
    amounts = [a[0] if isinstance(a, tuple) else a for a in amounts]
    has_url = bool(URL_RE.search(text))
    return {
        "tickers": tickers[:8],
        "hashtags": hashtags[:8],
        "addresses": list(addrs)[:8],
        "amounts": amounts[:8],
        "has_url": has_url,
    }


def plan_strings(plan: Dict[str, Any]) -> List[str]:
    out = []
    out += [f"${t}" for t in plan.get("tickers", [])]
    out += plan.get("hashtags", [])
    out += plan.get("addresses", [])
    out += plan.get("amounts", [])
    return [s for s in out if s]
