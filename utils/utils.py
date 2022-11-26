import random

import numpy as np
import torch


def fix_seed(seed: int):
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    # torch.use_deterministic_algorithms(mode=True, warn_only=True)
