"""Reproducibility: global seeds and DataLoader shuffling."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed random, numpy and torch (torch.manual_seed covers CPU, CUDA and MPS)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_generator(seed: int) -> torch.Generator:
    """Generator for DataLoader(shuffle=True, generator=...): reproducible batch order."""
    return torch.Generator().manual_seed(seed)
