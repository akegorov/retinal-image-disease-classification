"""Describe a training run for experiment logging: optimizer and model."""

from typing import Any

import torch
from torch import nn

OPTIMIZER_KEYS = (
    "lr",
    "betas",
    "momentum",
    "dampening",
    "nesterov",
    "alpha",
    "eps",
    "weight_decay",
    "amsgrad",
)


def optimizer_params(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Optimizer name and its main hyperparameters (OPTIMIZER_KEYS it has).

    betas are split into numbers optimizer/beta1 and optimizer/beta2.
    """
    params: dict[str, Any] = {"name": type(optimizer).__name__}
    for key in OPTIMIZER_KEYS:
        if key not in optimizer.defaults:
            continue
        if key == "betas":
            params["beta1"], params["beta2"] = map(float, optimizer.defaults[key])
        else:
            params[key] = optimizer.defaults[key]
    return {f"optimizer/{k}": v for k, v in params.items()}


def model_params(model: nn.Module) -> dict[str, Any]:
    """Model class and total / trainable parameter counts."""
    params = list(model.parameters())
    return {
        "model/class": type(model).__name__,
        "model/params_total": sum(p.numel() for p in params),
        "model/params_trainable": sum(p.numel() for p in params if p.requires_grad),
    }
