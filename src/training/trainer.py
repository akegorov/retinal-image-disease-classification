"""Generic multilabel training loop.

Models take the input tensor batch["x"] and return logits, so the trainer does not know
whether x is an embedding or an image.
"""

import copy
import time

import numpy as np
import torch
from torch import nn
from torch.nn.utils import get_total_norm
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.labels import LABELS
from src.loggers import Logger
from src.metrics.classification import compute_metrics
from src.training.run_info import model_params, optimizer_params


@torch.no_grad()
def _forward(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    show_batches: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Logits and labels (None if the batches have none) over the loader, on CPU."""
    model.to(device).eval()
    logits, labels = [], []
    for batch in tqdm(loader, desc="predict", leave=False, disable=not show_batches):
        logits.append(model(batch["x"].to(device)).float().cpu())
        labels.append(batch.get("labels"))
    return torch.cat(logits), None if labels[0] is None else torch.cat(labels)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Probabilities (n_samples, n_classes) in loader order."""
    return torch.sigmoid(_forward(model, loader, device)[0]).numpy()


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    logger: Logger,
    pos_weight: torch.Tensor | np.ndarray | None = None,
    weight_decay: float = 0.01,
    show_batches: bool = False,
    label_names: list[str] = LABELS,
) -> nn.Module:
    """Train with BCEWithLogitsLoss (+ optional pos_weight) and AdamW.

    Returns the model with the best val macro PR-AUC (better suited to rare classes
    than ROC-AUC).
    """
    if pos_weight is not None:
        pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    setup = optimizer_params(optimizer) | model_params(model)
    setup |= {
        "loss": type(criterion).__name__,
        "epochs": epochs,
        "train_batch_size": train_loader.batch_size,
        "val_batch_size": val_loader.batch_size,
        "n_train": len(train_loader.dataset),
        "n_val": len(val_loader.dataset),
    }
    if pos_weight is not None:
        setup |= {
            f"pos_weight/{name}": round(w, 4)
            for name, w in zip(label_names, pos_weight.tolist())
        }
    logger.log_params(setup)

    best_pr_auc, best_epoch, best_state = -np.inf, 0, None
    bar = tqdm(range(1, epochs + 1), desc="epochs")
    for epoch in bar:
        start = time.perf_counter()
        model.train()
        losses, grad_norms = [], []
        for batch in tqdm(train_loader, leave=False, disable=not show_batches):
            x, y = batch["x"].to(device), batch["labels"].to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            grad_norms.append(
                get_total_norm([p.grad for p in params if p.grad is not None]).item()
            )
            optimizer.step()
            losses.append(loss.item())

        logits, y_true = _forward(model, val_loader, device, show_batches)
        val = compute_metrics(
            y_true.numpy(), torch.sigmoid(logits).numpy(), label_names=label_names
        )
        metrics = {
            "train_loss": float(np.mean(losses)),
            "val_loss": criterion(logits.to(device), y_true.to(device)).item(),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": time.perf_counter() - start,
            "grad_norm": float(np.mean(grad_norms)),
        }
        metrics |= {
            f"val_{k}": v
            for k, v in val.items()
            if k.endswith("_macro") or k.startswith("roc_auc_")
        }
        logger.log_metrics(metrics, step=epoch, epoch=epoch)
        bar.set_postfix(
            train_loss=metrics["train_loss"], val_pr_auc=val["pr_auc_macro"]
        )

        if val["pr_auc_macro"] > best_pr_auc:
            best_pr_auc, best_epoch = val["pr_auc_macro"], epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.log_metrics({"best_epoch": best_epoch, "best_val_pr_auc_macro": best_pr_auc})
    return model
