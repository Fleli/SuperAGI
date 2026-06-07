from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from superagi.chat.sft import IGNORE_INDEX, TokenizedSftExample


@dataclass(frozen=True)
class SourceSummary:
    counts_text: str
    sampling_mass_text: str


def collate_sft_batch(
    examples: Sequence[TokenizedSftExample],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not examples:
        raise ValueError("at least one SFT example is required")

    max_length = max(len(example.input_ids) for example in examples)
    input_ids = torch.full(
        (len(examples), max_length),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    target_ids = torch.full(
        (len(examples), max_length),
        fill_value=IGNORE_INDEX,
        dtype=torch.long,
    )

    for row, example in enumerate(examples):
        if len(example.input_ids) != len(example.target_ids):
            raise ValueError("SFT input_ids and target_ids must have equal length")
        sequence_length = len(example.input_ids)
        input_ids[row, :sequence_length] = torch.tensor(
            example.input_ids,
            dtype=torch.long,
        )
        target_ids[row, :sequence_length] = torch.tensor(
            example.target_ids,
            dtype=torch.long,
        )

    return input_ids.to(device=device), target_ids.to(device=device)


def sample_sft_batch(
    examples: Sequence[TokenizedSftExample],
    *,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    source_weights: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not examples:
        raise ValueError("at least one SFT example is required")

    if source_weights:
        weights = source_sampling_weights(examples, source_weights)
        indices = torch.multinomial(
            weights,
            num_samples=batch_size,
            replacement=True,
            generator=generator,
        )
    else:
        indices = torch.randint(
            low=0,
            high=len(examples),
            size=(batch_size,),
            generator=generator,
        )
    batch_examples = [examples[int(index)] for index in indices.tolist()]
    return collate_sft_batch(
        batch_examples,
        pad_token_id=pad_token_id,
        device=device,
    )


def split_sft_examples(
    examples: Sequence[TokenizedSftExample],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[tuple[TokenizedSftExample, ...], tuple[TokenizedSftExample, ...]]:
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if not examples:
        raise ValueError("at least one SFT example is required")
    if validation_fraction == 0 or len(examples) < 2:
        return tuple(examples), ()

    validation_count = int(round(len(examples) * validation_fraction))
    validation_count = max(1, min(validation_count, len(examples) - 1))
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(examples), generator=generator).tolist()
    validation_indices = set(indices[:validation_count])
    train_examples = tuple(
        example for index, example in enumerate(examples) if index not in validation_indices
    )
    validation_examples = tuple(
        example for index, example in enumerate(examples) if index in validation_indices
    )
    return train_examples, validation_examples


def parse_sft_source_weights(value: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    if not value.strip():
        return weights

    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("source weights must use source=weight entries")
        source, raw_weight = item.split("=", maxsplit=1)
        source = source.strip()
        if not source:
            raise ValueError("source weight source name must be non-empty")
        try:
            weight = float(raw_weight)
        except ValueError as error:
            raise ValueError(f"source weight for {source!r} must be numeric") from error
        if weight < 0:
            raise ValueError(f"source weight for {source!r} must be non-negative")
        weights[source] = weight
    return weights


def source_sampling_weights(
    examples: Sequence[TokenizedSftExample],
    source_weights: Mapping[str, float],
) -> torch.Tensor:
    weights = torch.tensor(
        [
            _weight_for_source(example.source, source_weights)
            for example in examples
        ],
        dtype=torch.float,
    )
    if torch.sum(weights).item() <= 0:
        raise ValueError("at least one SFT source weight must be positive")
    return weights


def source_summary(
    examples: Sequence[TokenizedSftExample],
    source_weights: Mapping[str, float],
) -> SourceSummary:
    counts = Counter(_source_family(example.source) for example in examples)
    weighted_mass = {
        source: count * _weight_for_source(source, source_weights)
        for source, count in counts.items()
    }
    total_mass = sum(weighted_mass.values())
    counts_text = ", ".join(
        f"{source}={counts[source]}" for source in sorted(counts)
    )
    if total_mass <= 0:
        sampling_mass_text = "none"
    else:
        sampling_mass_text = ", ".join(
            f"{source}={(weighted_mass[source] / total_mass) * 100:.1f}%"
            for source in sorted(weighted_mass)
        )
    return SourceSummary(
        counts_text=counts_text,
        sampling_mass_text=sampling_mass_text,
    )


@torch.no_grad()
def evaluate_sft_loss(
    model: torch.nn.Module,
    examples: Sequence[TokenizedSftExample],
    *,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
    max_batches: int,
) -> float:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_batches <= 0:
        raise ValueError("max_batches must be positive")
    if not examples:
        raise ValueError("at least one SFT example is required")

    was_training = model.training
    model.eval()
    losses: list[float] = []
    for start in range(0, len(examples), batch_size):
        if len(losses) >= max_batches:
            break
        batch_examples = examples[start : start + batch_size]
        input_ids, target_ids = collate_sft_batch(
            batch_examples,
            pad_token_id=pad_token_id,
            device=device,
        )
        _, loss = model(input_ids, target_ids)
        if loss is None:
            raise RuntimeError("model did not return a validation loss")
        losses.append(float(loss.item()))
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def _weight_for_source(
    source: str,
    source_weights: Mapping[str, float],
) -> float:
    family = _source_family(source)
    if source in source_weights:
        return source_weights[source]
    if family in source_weights:
        return source_weights[family]
    return source_weights.get("default", 1.0)


def _source_family(source: str) -> str:
    return source.split(":", maxsplit=1)[0]
