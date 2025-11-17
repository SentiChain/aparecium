from transformers import AutoTokenizer


def build_tokenizer(name: str = "gpt2", max_len: int = 64):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.model_max_length = max_len
    return tok


def encode_batch(tokenizer, texts, max_len=64):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )


def decode_tokens(tokenizer, ids):
    return tokenizer.batch_decode(ids, skip_special_tokens=True)
