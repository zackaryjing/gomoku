from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gomoku.model import PolicyValueNet


@dataclass
class TrainConfig:
    batch_size: int = 256
    epochs: int = 1
    lr: float = 1e-3
    value_loss_weight: float = 1.0
    num_workers: int = 0


def load_npz_dataset(path: str | Path) -> TensorDataset:
    data = np.load(path)
    states = torch.from_numpy(data["states"]).float()
    policies = torch.from_numpy(data["policies"]).float()
    values = torch.from_numpy(data["values"]).float().unsqueeze(1)
    return TensorDataset(states, policies, values)


def train_model(
    model: PolicyValueNet,
    dataset: TensorDataset,
    config: TrainConfig,
    device: str = "cuda",
) -> dict[str, float]:
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

    for _ in range(config.epochs):
        model.train()
        for states, target_policy, target_value in loader:
            states = states.to(device)
            target_policy = target_policy.to(device)
            target_value = target_value.to(device)

            logits, value = model(states)
            policy_loss = soft_cross_entropy(logits, target_policy)
            value_loss = F.mse_loss(value, target_value)
            loss = policy_loss + config.value_loss_weight * value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            metrics = {
                "loss": float(loss.item()),
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
            }
    return metrics


def soft_cross_entropy(logits: torch.Tensor, target_policy: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(target_policy * log_probs).sum(dim=1).mean()
