import torch
from torch.utils.data import TensorDataset

from gomoku.model import PolicyValueNet
from gomoku.training import TrainConfig, load_checkpoint, move_optimizer_to_device, train_model


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


def test_move_optimizer_to_device_keeps_state_tensors_on_target(tmp_path):
    model = PolicyValueNet(channels=8, blocks=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    states = torch.zeros(2, 3, 15, 15)
    policies = torch.full((2, 225), 1.0 / 225.0)
    values = torch.zeros(2, 1)
    train_model(
        model,
        TensorDataset(states, policies, values),
        TrainConfig(batch_size=2, epochs=1),
        device="cpu",
        optimizer=optimizer,
    )

    move_optimizer_to_device(optimizer, "cpu")

    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                assert value.device.type == "cpu"
