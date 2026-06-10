"""
Helper functions
"""
import os
import random

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

def set_seed(random_seed: int):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


def run_name(batch_size, seed, metric=None):
    """Folder name identifying a run, e.g. 'bsz128_mahalanobis_seed42' or 'bsz128_seed42'."""
    parts = [f"bsz{batch_size}"]
    if metric is not None:
        parts.append(str(metric))
    parts.append(f"seed{seed}")
    return "_".join(parts)


def make_run_dir(root, batch_size, seed, metric=None):
    """Create <root>/<run_name> (if needed) and return the path."""
    path = os.path.join(root, run_name(batch_size, seed, metric=metric))
    os.makedirs(path, exist_ok=True)
    return path
