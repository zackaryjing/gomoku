from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from gomoku.model import PolicyValueNet


@dataclass
class TrainConfig:
    batch_size: int = 256
    epochs: int = 1
    lr: float = 1e-3
    value_loss_weight: float = 1.0
    num_workers: int = 0
    weight_decay: float = 1e-4
    save_every: int = 1


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    metrics: dict[str, float] | None = None


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
    optimizer: torch.optim.Optimizer | None = None,
    state: TrainState | None = None,
    checkpoint_path: str | Path | None = None,
) -> TrainState:
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    model.to(device)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    move_optimizer_to_device(optimizer, device)
    train_state = state or TrainState()
    metrics = train_state.metrics or {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

    # Epoch numbers in checkpoints are completed epochs, so resume starts at the next one.
    target_epoch = train_state.epoch + config.epochs
    epoch_iter = tqdm(
        range(train_state.epoch, target_epoch),
        initial=train_state.epoch,
        total=target_epoch,
        desc="epochs",
        unit="epoch",
    )
    for epoch in epoch_iter:
        model.train()
        batch_iter = tqdm(loader, desc=f"epoch {epoch + 1}", unit="batch", leave=False)
        for states, target_policy, target_value in batch_iter:
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
            train_state.global_step += 1
            train_state.metrics = metrics
            batch_iter.set_postfix(metrics)

        train_state.epoch = epoch + 1
        epoch_iter.set_postfix(metrics)
        if checkpoint_path and config.save_every > 0 and train_state.epoch % config.save_every == 0:
            save_checkpoint(checkpoint_path, model, optimizer, train_state, config)

    return train_state


def save_checkpoint(
    path: str | Path,
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    state: TrainState,
    config: TrainConfig,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": state.epoch,
            "global_step": state.global_step,
            "metrics": state.metrics or {},
            "config": config.__dict__,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> TrainState:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model"])

    # Older checkpoints may only contain model weights and metrics.
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return TrainState(
        epoch=int(checkpoint.get("epoch", 0)),
        global_step=int(checkpoint.get("global_step", 0)),
        metrics=dict(checkpoint.get("metrics", {})),
    )


def move_optimizer_to_device(optimizer: torch.optim.Optimizer, device: str | torch.device) -> None:
    target = torch.device(device)
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(target)


def soft_cross_entropy(logits: torch.Tensor, target_policy: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(target_policy * log_probs).sum(dim=1).mean()
