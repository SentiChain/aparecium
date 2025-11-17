from typing import Dict, List, Optional
import torch, torch.nn.functional as F


def _deterministic_argsort(
    values: torch.Tensor, ids: torch.Tensor, topk: int
) -> torch.Tensor:
    """Return indices that sort by values desc, then by ids lexicographically as tie-break.
    values: (N,), ids: (N, T)
    """
    # Primary sort by values
    order = torch.argsort(values, descending=True)
    # Stable tie-break by ids (lexicographic on token ids)
    # We implement a simple pass: for equal values within float tolerance, sort by ids.
    if order.numel() <= 1:
        return order[:topk]
    selected = []
    i = 0
    eps = 1e-9
    while i < order.numel():
        j = i + 1
        while (
            j < order.numel()
            and abs(values[order[j]].item() - values[order[i]].item()) <= eps
        ):
            j += 1
        group = order[i:j]
        if group.numel() > 1:
            # Sort group by ids lexicographically
            # Convert to list of tuples of ints for Python sort stability
            tuples = [(ids[idx].tolist(), idx.item()) for idx in group]
            tuples.sort(key=lambda x: x[0])
            group = torch.tensor(
                [t[1] for t in tuples], device=order.device, dtype=order.dtype
            )
        selected.append(group)
        i = j
    order2 = torch.cat(selected, dim=0)
    return order2[:topk]


def deterministic_beam_search(
    decoder,
    tokenizer,
    H,
    beam: int = 5,
    max_len: int = 64,
    rnet: Optional[object] = None,
    e: Optional[torch.Tensor] = None,
    rerank_every: int = 8,
    constraints: Optional[List[object]] = None,
    alpha: float = 1.0,
) -> Dict[str, object]:
    """Run deterministic beam search with optional surrogate reranking and constraints.

    Returns dict with texts, ids, logp (LM), lengths.
    """
    device = H.device
    B = H.size(0)
    # Choose a safe BOS token. If tokenizer has no BOS, prefer a plain space token
    # over EOS to reduce degenerate generations when starting decoding.
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None:
        try:
            toks = tokenizer(" ", add_special_tokens=False)
            ids0 = toks["input_ids"] if isinstance(toks, dict) else toks
            bos = (
                ids0[0]
                if isinstance(ids0, list) and len(ids0) > 0
                else tokenizer.eos_token_id
            )
        except Exception:
            bos = tokenizer.eos_token_id
    eos = tokenizer.eos_token_id
    beams = torch.full((B, beam, 1), bos, dtype=torch.long, device=device)
    scores = torch.zeros(B, beam, device=device)

    for t in range(1, max_len):
        # Expand each beam independently
        all_candidates = []
        for b in range(beam):
            logits = decoder(beams[:, b, :], memory=H)[:, -1, :]  # (B, V)
            logp = F.log_softmax(logits, dim=-1)
            topk = torch.topk(logp, k=beam, dim=-1)  # (B, beam)
            # Create candidates
            for k in range(beam):
                next_id = topk.indices[:, k].unsqueeze(-1)  # (B,1)
                next_score = scores[:, b] + topk.values[:, k]
                next_seq = torch.cat([beams[:, b, :], next_id], dim=-1)
                all_candidates.append((next_seq, next_score))

        # Select top candidates by LM score (initial prune)
        seqs = torch.stack([c[0] for c in all_candidates], dim=1)  # (B, beam*beam, T)
        scs = torch.stack([c[1] for c in all_candidates], dim=1)  # (B, beam*beam)

        new_beams = []
        new_scores = []
        for i in range(B):
            # First prune to beam by LM scores with deterministic tie-break
            ids_i = seqs[i]  # (beam*beam, T)
            vals_i = scs[i]  # (beam*beam)
            idx_i = _deterministic_argsort(vals_i, ids_i, topk=beam)
            beams_i = ids_i[idx_i]
            scores_i = vals_i[idx_i]

            # Optional rerank by surrogate r and constraints
            need_rerank = (
                rnet is not None and e is not None and (t % max(1, rerank_every) == 0)
            )
            if need_rerank:
                texts = tokenizer.batch_decode(
                    beams_i.tolist(), skip_special_tokens=True
                )
                # Surrogate r(x,e)
                toks = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                toks = {k: v.to(device) for k, v in toks.items()}
                e_i = e[i].unsqueeze(0).repeat(beams_i.size(0), 1).to(device)
                with torch.no_grad():
                    r_scores = rnet(
                        toks["input_ids"], toks.get("attention_mask", None), e_i
                    )
                fused = scores_i.clone()
                fused = fused + alpha * r_scores
                # Constraints penalty (if provided)
                if (
                    constraints is not None
                    and i < len(constraints)
                    and constraints[i] is not None
                ):
                    try:
                        from ..models.constraints import apply_constraints_penalty

                        penalties = apply_constraints_penalty(
                            texts, [constraints[i]]
                        ).to(device)
                        fused = fused + penalties
                    except Exception:
                        pass
                # Reorder by fused, keeping LM scores for output
                order = _deterministic_argsort(fused, beams_i, topk=beam)
                beams_i = beams_i[order]
                scores_i = scores_i[order]

            new_beams.append(beams_i)
            new_scores.append(scores_i)

        beams = torch.stack(new_beams, dim=0)
        scores = torch.stack(new_scores, dim=0)

        # Early stop if all have EOS
        if (beams[:, :, -1] == eos).all():
            break

    # Final rerank at end if not already on a rerank step
    if rnet is not None and e is not None:
        new_beams = []
        new_scores = []
        for i in range(B):
            texts = tokenizer.batch_decode(beams[i].tolist(), skip_special_tokens=True)
            toks = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            toks = {k: v.to(device) for k, v in toks.items()}
            e_i = e[i].unsqueeze(0).repeat(beams.size(1), 1).to(device)
            with torch.no_grad():
                r_scores = rnet(
                    toks["input_ids"], toks.get("attention_mask", None), e_i
                )
            fused = scores[i].clone() + alpha * r_scores
            # Constraints penalty (if provided)
            if (
                constraints is not None
                and i < len(constraints)
                and constraints[i] is not None
            ):
                try:
                    from ..models.constraints import apply_constraints_penalty

                    penalties = apply_constraints_penalty(texts, [constraints[i]]).to(
                        device
                    )
                    fused = fused + penalties
                except Exception:
                    pass
            order = _deterministic_argsort(fused, beams[i], topk=beam)
            new_beams.append(beams[i][order])
            new_scores.append(scores[i][order])
        beams = torch.stack(new_beams, dim=0)
        scores = torch.stack(new_scores, dim=0)

    # Decode beams to text
    ids = beams
    texts = []
    lengths = []
    for i in range(B):
        row_texts = tokenizer.batch_decode(ids[i].tolist(), skip_special_tokens=True)
        texts.append(row_texts)
        lengths.append(torch.tensor([len(t.strip().split()) for t in row_texts]))
    lengths = torch.stack(lengths, dim=0).to(H.device)
    return {"texts": texts, "ids": ids, "logp": scores, "lengths": lengths}
