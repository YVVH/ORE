import os

import torch

if torch.cuda.is_available():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    DEVICE = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(DEVICE)
else:
    DEVICE = torch.device("cpu")


def empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
