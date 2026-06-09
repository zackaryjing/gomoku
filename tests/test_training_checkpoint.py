import torch
from torch.utils.data import TensorDataset

from gomoku.model import PolicyValueNet
from gomoku.training import TrainConfig, load_checkpoint, train_model


def test_training_checkpoint_resume(tmp_path):
    states = torch.zeros(4, 3, 15, 15)
    policies = torch.full((4, 225), 1.0 / 225.0)
    values = torch.zeros(4, 1)
    dataset = TensorDataset(states, policies, values)
    checkpoint_path = tmp_path / "resume.pt"

    model = PolicyValueNet(channels=8, blocks=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    state = train_model(
        model,
        dataset,
        TrainConfig(batch_size=2, epochs=1),
        device="cpu",
        optimizer=optimizer,
        checkpoint_path=checkpoint_path,
    )

    resumed_model = PolicyValueNet(channels=8, blocks=1)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_state = load_checkpoint(checkpoint_path, resumed_model, resumed_optimizer)

    assert state.epoch == 1
    assert resumed_state.epoch == 1
    assert resumed_state.global_step == 2
    assert "loss" in (resumed_state.metrics or {})
