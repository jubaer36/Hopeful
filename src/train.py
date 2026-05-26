"""Training and evaluation loops."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import compute_metrics


def _to_device(conv: dict, device: torch.device) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in conv.items()
    }


def list_collate(batch):
    """DataLoader collate: return conversations as a plain list (no padding)."""
    return batch


def train_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    epoch: int,
    device: torch.device,
    grad_clip: float,
    num_classes: int,
) -> dict:
    model.train()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for batch in loader:
        batch = [_to_device(c, device) for c in batch]
        logits, labels = model(batch)
        loss = loss_fn(logits, labels, epoch)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_preds, all_labels, num_classes)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    loss_fn,
    epoch: int,
    device: torch.device,
    num_classes: int,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for batch in loader:
        batch = [_to_device(c, device) for c in batch]
        logits, labels = model(batch)
        loss = loss_fn(logits, labels, epoch)

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_preds, all_labels, num_classes)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics
