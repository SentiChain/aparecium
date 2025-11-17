from typing import List


def normalize_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    return s


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_text(pred) == normalize_text(gold))


def jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    return len(A & B) / max(1, len(A | B))


def numbers_from(s: str) -> List[str]:
    return re.findall(r"\d+(?:\.\d+)?%?", s)


def number_accuracy(pred: str, gold: str) -> float:
    pa, ga = numbers_from(pred), numbers_from(gold)
    return jaccard(pa, ga)
