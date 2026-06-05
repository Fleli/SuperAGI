from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class NextTokenDataset(Dataset):
    def __init__(self, token_ids: Sequence[int], context_length: int) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if len(token_ids) <= context_length:
            raise ValueError("token_ids must be longer than context_length")

        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.tokens) - self.context_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index
        end = start + self.context_length
        input_ids = self.tokens[start:end]
        target_ids = self.tokens[start + 1 : end + 1]
        return input_ids, target_ids


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_steps: int = 1000
    weight_decay: float = 0.01
    grad_clip: float | None = 1.0
    shuffle: bool = True

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive when set")


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    grad_clip: float | None = 1.0,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(input_ids, target_ids)
    if loss is None:
        raise RuntimeError("model did not return a loss")

    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(loss.detach().item())


def train_model(
    model: nn.Module,
    token_ids: Sequence[int],
    config: TrainConfig,
    device: torch.device | str = "cpu",
) -> list[float]:
    device = torch.device(device)
    model.to(device)
    dataset = NextTokenDataset(
        token_ids=token_ids,
        context_length=model.config.context_length,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    losses = []
    batches = cycle(data_loader)
    for _ in range(config.max_steps):
        input_ids, target_ids = next(batches)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        losses.append(
            train_step(
                model=model,
                optimizer=optimizer,
                input_ids=input_ids,
                target_ids=target_ids,
                grad_clip=config.grad_clip,
            )
        )
    return losses
