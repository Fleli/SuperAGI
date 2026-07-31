from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from superagi.chat.sft import IGNORE_INDEX, TokenizedSftExample
from superagi.ingestion.tokenizer import PAD_TOKEN, TokenizerLike


@dataclass(frozen=True)
class SourceSummary:
    counts_text: str
    sampling_mass_text: str


def resolve_sft_pad_token_id(tokenizer: TokenizerLike) -> int:
    try:
        return int(tokenizer.special_token_id(PAD_TOKEN))
    except (AttributeError, ValueError) as error:
        raise ValueError("SFT tokenizer must define a <pad> special token") from error


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

    generator = torch.Generator()
    generator.manual_seed(seed)

    effective_group_keys = tuple(
        example.group_key or f"__legacy_sft_example_{index}__"
        for index, example in enumerate(examples)
    )
    groups_by_source: dict[
        str,
        dict[str, list[TokenizedSftExample]],
    ] = {}
    for example, group_key in zip(examples, effective_group_keys, strict=True):
        source_groups = groups_by_source.setdefault(_source_family(example.source), {})
        source_groups.setdefault(group_key, []).append(example)

    validation_groups: set[tuple[str, str]] = set()
    train_group_keys: set[str] = set()
    validation_group_keys: set[str] = set()
    for source_family in sorted(groups_by_source):
        source_groups = groups_by_source[source_family]
        source_example_count = sum(len(group) for group in source_groups.values())
        if source_example_count < 2:
            for group_key in source_groups:
                if group_key in validation_group_keys:
                    validation_groups.add((source_family, group_key))
                else:
                    train_group_keys.add(group_key)
            continue

        validation_target = int(round(source_example_count * validation_fraction))
        validation_target = max(1, min(validation_target, source_example_count - 1))
        group_keys = list(source_groups)
        shuffled_indices = torch.randperm(len(group_keys), generator=generator).tolist()
        validation_count = 0
        for index in shuffled_indices:
            group_key = group_keys[index]
            group_size = len(source_groups[group_key])
            if group_key in validation_group_keys:
                validation_groups.add((source_family, group_key))
                validation_count += group_size
                continue
            if group_key in train_group_keys:
                continue
            if (
                validation_count < validation_target
                and validation_count + group_size < source_example_count
            ):
                validation_groups.add((source_family, group_key))
                validation_group_keys.add(group_key)
                validation_count += group_size
            else:
                train_group_keys.add(group_key)

    train_examples = tuple(
        example
        for example, group_key in zip(examples, effective_group_keys, strict=True)
        if (_source_family(example.source), group_key) not in validation_groups
    )
    validation_examples = tuple(
        example
        for example, group_key in zip(examples, effective_group_keys, strict=True)
        if (_source_family(example.source), group_key) in validation_groups
    )
    validation_examples = _interleave_validation_sources(
        validation_examples,
        seed=seed,
    )
    train_group_keys = {
        group_key
        for example, group_key in zip(examples, effective_group_keys, strict=True)
        if (_source_family(example.source), group_key) not in validation_groups
    }
    validation_group_keys = {
        group_key
        for example, group_key in zip(examples, effective_group_keys, strict=True)
        if (_source_family(example.source), group_key) in validation_groups
    }
    if train_group_keys & validation_group_keys:
        raise AssertionError("SFT train and validation group keys must be disjoint")
    return train_examples, validation_examples


def _interleave_validation_sources(
    examples: Sequence[TokenizedSftExample],
    *,
    seed: int,
) -> tuple[TokenizedSftExample, ...]:
    """Build a deterministic source-balanced order for bounded validation."""
    if len(examples) < 2:
        return tuple(examples)

    by_source: dict[str, list[TokenizedSftExample]] = {}
    for example in examples:
        by_source.setdefault(_source_family(example.source), []).append(example)

    generator = torch.Generator()
    generator.manual_seed(seed + 1)
    shuffled_by_source: dict[str, list[TokenizedSftExample]] = {}
    for source in sorted(by_source):
        source_examples = by_source[source]
        indices = torch.randperm(len(source_examples), generator=generator).tolist()
        shuffled_by_source[source] = [source_examples[index] for index in indices]

    ordered: list[TokenizedSftExample] = []
    sources = tuple(sorted(shuffled_by_source))
    offset = 0
    while len(ordered) < len(examples):
        for source in sources:
            source_examples = shuffled_by_source[source]
            if offset < len(source_examples):
                ordered.append(source_examples[offset])
        offset += 1
    return tuple(ordered)


def limit_sft_examples(
    examples: Sequence[TokenizedSftExample],
    *,
    max_examples: int,
    seed: int,
) -> tuple[TokenizedSftExample, ...]:
    if max_examples < 0:
        raise ValueError("max_examples must be non-negative")
    if not examples:
        raise ValueError("at least one SFT example is required")
    if max_examples == 0 or len(examples) <= max_examples:
        return tuple(examples)

    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(examples), generator=generator)[:max_examples]
    return tuple(examples[int(index)] for index in indices.tolist())


def should_log_sft_progress(
    *,
    step: int,
    total_steps: int,
    log_interval: int,
    checkpoint_interval: int,
) -> bool:
    if step <= 0:
        return False
    if step == total_steps:
        return True
    if log_interval > 0:
        return step % log_interval == 0
    return checkpoint_interval > 0 and step % checkpoint_interval == 0


def clear_sft_device_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        return
    if device.type == "mps":
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache"):
            mps.empty_cache()


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
    mixed_precision_dtype: torch.dtype | None = None,
) -> float:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_batches <= 0:
        raise ValueError("max_batches must be positive")
    if not examples:
        raise ValueError("at least one SFT example is required")

    was_training = model.training
    model.eval()
    total_weighted_loss = 0.0
    total_supervised_tokens = 0
    batch_count = 0
    for start in range(0, len(examples), batch_size):
        if batch_count >= max_batches:
            break
        batch_examples = examples[start : start + batch_size]
        input_ids, target_ids = collate_sft_batch(
            batch_examples,
            pad_token_id=pad_token_id,
            device=device,
        )
        with torch.amp.autocast(
            device_type=device.type,
            dtype=mixed_precision_dtype,
            enabled=mixed_precision_dtype is not None,
        ):
            _, loss = model(input_ids, target_ids)
        if loss is None:
            raise RuntimeError("model did not return a validation loss")
        supervised_tokens = int(target_ids.ne(IGNORE_INDEX).sum().item())
        if supervised_tokens == 0:
            raise RuntimeError("SFT validation batch has no supervised tokens")
        total_weighted_loss += float(loss.item()) * supervised_tokens
        total_supervised_tokens += supervised_tokens
        batch_count += 1
    if was_training:
        model.train()
    if total_supervised_tokens == 0:
        raise RuntimeError("SFT validation produced no supervised tokens")
    return total_weighted_loss / total_supervised_tokens


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
