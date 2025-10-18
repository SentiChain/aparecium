from typing import List, Dict, Any, Tuple
import torch


class MustIncludeStrings:
    """Naive constraint: ensure certain substrings appear in final decoded text.
    We don't hard-mask tokens (complex for BPE); instead we add a large penalty
    if, at EOS, required strings are missing. For mid-decoding pruning,
    we apply a light heuristic: if the partial hypothesis length exceeds max_len-remaining-needed,
    favor continuations that include required prefixes.
    """

    def __init__(self, required_strings: List[str]):
        self.required = [s for s in required_strings if s]

    def penalty(self, text: str) -> float:
        miss = 0
        for s in self.required:
            if s not in text:
                miss += 1
        # return negative penalty to add to scores
        return -5.0 * miss  # tune


def apply_constraints_penalty(
    texts: List[str], constraints: List[MustIncludeStrings]
) -> torch.Tensor:
    if not constraints:
        return torch.zeros(len(texts))
    total = []
    for t in texts:
        pen = 0.0
        for c in constraints:
            pen += c.penalty(t)
        total.append(pen)
    return torch.tensor(total, dtype=torch.float32)
