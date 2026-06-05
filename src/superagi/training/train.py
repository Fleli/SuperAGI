from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch import nn
from torch.utils.data import Dataset


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


@dataclass(frozen=True)
class MetricSnapshot:
    step: int
    train_loss: float
    validation_loss: float | None
    elapsed_seconds: float


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


def sample_next_token_batch(
    *,
    tokens: torch.Tensor,
    context_length: int,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.ndim != 1:
        raise ValueError("tokens must be a 1D tensor")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(tokens) <= context_length:
        raise ValueError("tokens must be longer than context_length")

    starts = torch.randint(
        low=0,
        high=len(tokens) - context_length,
        size=(batch_size,),
        device=tokens.device,
        generator=generator,
    )
    offsets = torch.arange(context_length, device=tokens.device)
    input_positions = starts[:, None] + offsets[None, :]
    target_positions = input_positions + 1
    return tokens[input_positions], tokens[target_positions]


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    token_ids: Sequence[int],
    batch_size: int,
    max_batches: int,
    device: torch.device | str = "cpu",
) -> float:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_batches <= 0:
        raise ValueError("max_batches must be positive")

    device = torch.device(device)
    tokens = _tokens_to_device(token_ids, device)
    was_training = model.training
    model.eval()

    losses = []
    for batch_index in range(max_batches):
        start = batch_index * batch_size
        if start >= len(tokens) - model.config.context_length:
            break
        end = min(start + batch_size, len(tokens) - model.config.context_length)
        starts = torch.arange(start, end, device=device)
        offsets = torch.arange(model.config.context_length, device=device)
        input_positions = starts[:, None] + offsets[None, :]
        target_positions = input_positions + 1
        input_ids = tokens[input_positions]
        target_ids = tokens[target_positions]
        _, loss = model(input_ids, target_ids)
        if loss is None:
            raise RuntimeError("model did not return a loss")
        losses.append(float(loss.detach().item()))

    if was_training:
        model.train()
    return sum(losses) / len(losses)


def train_model(
    model: nn.Module,
    token_ids: Sequence[int],
    config: TrainConfig,
    device: torch.device | str = "cpu",
) -> list[float]:
    losses, _ = train_model_with_metrics(
        model=model,
        token_ids=token_ids,
        config=config,
        device=device,
    )
    return losses


def train_model_with_metrics(
    model: nn.Module,
    token_ids: Sequence[int],
    config: TrainConfig,
    device: torch.device | str = "cpu",
    validation_token_ids: Sequence[int] | None = None,
    eval_interval: int = 0,
    validation_batches: int = 10,
    start_step: int = 0,
    checkpoint_interval: int = 0,
    checkpoint_callback: Callable[[int, list[float], list[MetricSnapshot]], None] | None = None,
    metric_callback: Callable[[MetricSnapshot, list[float], list[MetricSnapshot]], None] | None = None,
) -> tuple[list[float], list[MetricSnapshot]]:
    if eval_interval < 0:
        raise ValueError("eval_interval must be non-negative")
    if validation_batches <= 0:
        raise ValueError("validation_batches must be positive")
    if start_step < 0:
        raise ValueError("start_step must be non-negative")
    if checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be non-negative")

    device = torch.device(device)
    model.to(device)
    tokens = _tokens_to_device(token_ids, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    losses = []
    metrics = []
    start_time = time.perf_counter()
    for step_index in range(1, config.max_steps + 1):
        input_ids, target_ids = sample_next_token_batch(
            tokens=tokens,
            context_length=model.config.context_length,
            batch_size=config.batch_size,
        )
        train_loss = train_step(
            model=model,
            optimizer=optimizer,
            input_ids=input_ids,
            target_ids=target_ids,
            grad_clip=config.grad_clip,
        )
        losses.append(train_loss)

        should_eval = eval_interval > 0 and (
            step_index % eval_interval == 0 or step_index == config.max_steps
        )
        if should_eval:
            validation_loss = None
            if validation_token_ids is not None:
                validation_loss = evaluate_loss(
                    model=model,
                    token_ids=validation_token_ids,
                    batch_size=config.batch_size,
                    max_batches=validation_batches,
                    device=device,
                )
            metrics.append(
                MetricSnapshot(
                    step=start_step + step_index,
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                    elapsed_seconds=time.perf_counter() - start_time,
                )
            )
            print(
                "step="
                f"{metrics[-1].step} "
                f"train_loss={metrics[-1].train_loss:.6f} "
                f"validation_loss={_format_loss(metrics[-1].validation_loss)} "
                f"elapsed_seconds={metrics[-1].elapsed_seconds:.2f}",
                flush=True,
            )
            if metric_callback is not None:
                metric_callback(metrics[-1], list(losses), list(metrics))
        should_checkpoint = (
            checkpoint_callback is not None
            and checkpoint_interval > 0
            and (step_index % checkpoint_interval == 0 or step_index == config.max_steps)
        )
        if should_checkpoint:
            checkpoint_callback(start_step + step_index, list(losses), list(metrics))
    return losses, metrics


def append_metrics_jsonl(
    path: Path | str,
    metrics: Sequence[MetricSnapshot],
) -> None:
    if not metrics:
        return

    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as file:
        for metric in metrics:
            file.write(json.dumps(asdict(metric), sort_keys=True))
            file.write("\n")


def _tokens_to_device(
    token_ids: Sequence[int] | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(token_ids, torch.Tensor):
        return token_ids.to(device=device, dtype=torch.long)
    return torch.tensor(token_ids, dtype=torch.long, device=device)


def _format_loss(loss: float | None) -> str:
    if loss is None:
        return "null"
    return f"{loss:.6f}"
