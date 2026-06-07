from __future__ import annotations

import json
import math
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
class TokenShard:
    path: Path
    token_count: int


class TokenShardDataset:
    def __init__(self, shards: Sequence[TokenShard]) -> None:
        if not shards:
            raise ValueError("at least one token shard is required")
        self.shards = tuple(shards)
        self.total_tokens = sum(shard.token_count for shard in self.shards)
        self._loaded_path: Path | None = None
        self._loaded_device: torch.device | None = None
        self._loaded_tokens: torch.Tensor | None = None

    @classmethod
    def from_manifest(cls, manifest_path: Path | str) -> "TokenShardDataset":
        path = Path(manifest_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "superagi-token-shards-v1":
            raise ValueError("token shard manifest has unknown format")
        shard_records = payload.get("shards")
        if not isinstance(shard_records, list):
            raise ValueError("token shard manifest must contain shards")

        shards = []
        for record in shard_records:
            if not isinstance(record, dict):
                raise ValueError("token shard records must be mappings")
            relative_path = record.get("path")
            token_count = record.get("tokens")
            if not isinstance(relative_path, str):
                raise ValueError("token shard path must be a string")
            if not isinstance(token_count, int) or token_count <= 0:
                raise ValueError("token shard token count must be positive")
            shards.append(
                TokenShard(
                    path=path.parent / relative_path,
                    token_count=token_count,
                )
            )
        return cls(shards)

    @property
    def shard_count(self) -> int:
        return len(self.shards)

    def sample_next_token_batch(
        self,
        *,
        context_length: int,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eligible_shards = [
            shard for shard in self.shards if shard.token_count > context_length
        ]
        if not eligible_shards:
            raise ValueError("all token shards are shorter than context_length")

        selected_index = int(torch.randint(len(eligible_shards), size=(1,)).item())
        selected_shard = eligible_shards[selected_index]
        tokens = self._load_shard(selected_shard.path, device)
        return sample_next_token_batch(
            tokens=tokens,
            context_length=context_length,
            batch_size=batch_size,
        )

    def _load_shard(self, path: Path, device: torch.device) -> torch.Tensor:
        if (
            self._loaded_tokens is not None
            and self._loaded_path == path
            and self._loaded_device == device
        ):
            return self._loaded_tokens

        tokens = torch.load(path, map_location="cpu")
        if not isinstance(tokens, torch.Tensor):
            raise ValueError(f"token shard must contain a tensor: {path}")
        if tokens.ndim != 1:
            raise ValueError(f"token shard must contain a 1D tensor: {path}")
        self._loaded_path = path
        self._loaded_device = device
        self._loaded_tokens = tokens.to(device=device, dtype=torch.long)
        return self._loaded_tokens


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    min_learning_rate: float = 0.0
    warmup_steps: int = 0
    max_steps: int = 1000
    weight_decay: float = 0.01
    grad_clip: float | None = 1.0
    shuffle: bool = True
    mixed_precision: str = "none"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.min_learning_rate < 0:
            raise ValueError("min_learning_rate must be non-negative")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must be less than or equal to learning_rate")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive when set")
        if self.mixed_precision not in {"none", "auto", "float16", "bfloat16"}:
            raise ValueError("mixed_precision must be one of: none, auto, float16, bfloat16")


@dataclass(frozen=True)
class MetricSnapshot:
    step: int
    train_loss: float
    validation_loss: float | None
    learning_rate: float
    elapsed_seconds: float


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    grad_clip: float | None = 1.0,
    mixed_precision_dtype: torch.dtype | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(
        device_type=input_ids.device.type,
        dtype=mixed_precision_dtype,
        enabled=mixed_precision_dtype is not None,
    ):
        _, loss = model(input_ids, target_ids)
    if loss is None:
        raise RuntimeError("model did not return a loss")

    if grad_scaler is not None and grad_scaler.is_enabled():
        grad_scaler.scale(loss).backward()
        if grad_clip is not None:
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
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
    mixed_precision_dtype: torch.dtype | None = None,
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
        with torch.amp.autocast(
            device_type=device.type,
            dtype=mixed_precision_dtype,
            enabled=mixed_precision_dtype is not None,
        ):
            _, loss = model(input_ids, target_ids)
        if loss is None:
            raise RuntimeError("model did not return a loss")
        losses.append(float(loss.detach().item()))

    if was_training:
        model.train()
    return sum(losses) / len(losses)


def train_model(
    model: nn.Module,
    token_ids: Sequence[int] | torch.Tensor | TokenShardDataset,
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
    token_ids: Sequence[int] | torch.Tensor | TokenShardDataset,
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
    tokens = None if isinstance(token_ids, TokenShardDataset) else _tokens_to_device(token_ids, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    mixed_precision_dtype = _resolve_mixed_precision_dtype(
        config.mixed_precision,
        device,
    )
    grad_scaler = _build_grad_scaler(mixed_precision_dtype, device)

    losses = []
    metrics = []
    start_time = time.perf_counter()
    for step_index in range(1, config.max_steps + 1):
        current_learning_rate = learning_rate_for_step(config, step_index)
        _set_optimizer_learning_rate(optimizer, current_learning_rate)
        if isinstance(token_ids, TokenShardDataset):
            input_ids, target_ids = token_ids.sample_next_token_batch(
                context_length=model.config.context_length,
                batch_size=config.batch_size,
                device=device,
            )
        else:
            if tokens is None:
                raise RuntimeError("tokens were not initialized")
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
            mixed_precision_dtype=mixed_precision_dtype,
            grad_scaler=grad_scaler,
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
                    mixed_precision_dtype=mixed_precision_dtype,
                )
            metrics.append(
                MetricSnapshot(
                    step=start_step + step_index,
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                    learning_rate=current_learning_rate,
                    elapsed_seconds=time.perf_counter() - start_time,
                )
            )
            print(
                "step="
                f"{metrics[-1].step} "
                f"train_loss={metrics[-1].train_loss:.6f} "
                f"validation_loss={_format_loss(metrics[-1].validation_loss)} "
                f"learning_rate={metrics[-1].learning_rate:.8f} "
                f"elapsed_seconds={metrics[-1].elapsed_seconds:.2f} "
                f"estimated_remaining_seconds={_estimate_remaining_seconds(metrics[-1].elapsed_seconds, step_index, config.max_steps):.2f}",
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


def learning_rate_for_step(config: TrainConfig, step_index: int) -> float:
    if step_index <= 0:
        raise ValueError("step_index must be positive")
    if config.max_steps <= 0:
        return config.learning_rate

    effective_step = min(step_index, config.max_steps)
    warmup_steps = min(config.warmup_steps, config.max_steps)
    if warmup_steps > 0 and effective_step <= warmup_steps:
        return config.learning_rate * (effective_step / warmup_steps)

    decay_steps = config.max_steps - warmup_steps
    if decay_steps <= 0:
        return config.learning_rate

    decay_step = min(decay_steps, max(0, effective_step - warmup_steps))
    progress = decay_step / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + (
        config.learning_rate - config.min_learning_rate
    ) * cosine


def _set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


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


def _estimate_remaining_seconds(
    elapsed_seconds: float,
    completed_steps: int,
    total_steps: int,
) -> float:
    if completed_steps <= 0:
        return 0.0
    remaining_steps = max(0, total_steps - completed_steps)
    return (elapsed_seconds / completed_steps) * remaining_steps


def _resolve_mixed_precision_dtype(
    mixed_precision: str,
    device: torch.device,
) -> torch.dtype | None:
    if mixed_precision == "none":
        return None
    if mixed_precision == "auto":
        return torch.float16 if device.type == "cuda" else None
    if mixed_precision == "float16":
        return torch.float16
    if mixed_precision == "bfloat16":
        return torch.bfloat16
    raise ValueError("mixed_precision must be one of: none, auto, float16, bfloat16")


def _build_grad_scaler(
    mixed_precision_dtype: torch.dtype | None,
    device: torch.device,
) -> torch.amp.GradScaler | None:
    if mixed_precision_dtype != torch.float16 or device.type != "cuda":
        return None
    return torch.amp.GradScaler(device.type, enabled=True)
