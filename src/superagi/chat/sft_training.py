from __future__ import annotations

from collections.abc import Sequence

import torch

from superagi.chat.sft import IGNORE_INDEX, TokenizedSftExample


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
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not examples:
        raise ValueError("at least one SFT example is required")

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
