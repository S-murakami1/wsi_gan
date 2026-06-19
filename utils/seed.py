import random

import numpy as np
import torch
from loguru import logger


def set_seed(seed: int | None, *, deterministic: bool = True) -> None:
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info("seed set | seed={} | deterministic={}", seed, deterministic)
