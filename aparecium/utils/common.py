import random, numpy as np, torch


def set_determinism(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(x, device) for x in batch)
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.cnt += n

    @property
    def avg(self):
        return self.sum / max(1, self.cnt)
