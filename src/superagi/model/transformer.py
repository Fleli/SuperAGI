from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from superagi.model.embedding import TokenAndPositionEmbedding
from superagi.model.inference.attend import CausalSelfAttention
from superagi.model.inference.percept import FeedForward


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    context_length: int
    dim_embedding: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dim_feed_forward: int | None = None
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.dim_embedding <= 0:
            raise ValueError("dim_embedding must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.dim_embedding % self.n_heads != 0:
            raise ValueError("dim_embedding must be divisible by n_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0)")
        if self.dim_feed_forward is None:
            object.__setattr__(self, "dim_feed_forward", 4 * self.dim_embedding)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.dim_embedding)
        self.attention = CausalSelfAttention(
            dim_embedding=config.dim_embedding,
            n_heads=config.n_heads,
            context_length=config.context_length,
            dropout=config.dropout,
        )
        self.ln2 = nn.LayerNorm(config.dim_embedding)
        self.feed_forward = FeedForward(
            dim_embedding=config.dim_embedding,
            dim_feed_forward=config.dim_feed_forward,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = TokenAndPositionEmbedding(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            dim_embedding=config.dim_embedding,
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.ln_final = nn.LayerNorm(config.dim_embedding)
        self.output_projection = nn.Linear(
            config.dim_embedding,
            config.vocab_size,
            bias=False,
        )
        self.output_projection.weight = self.embeddings.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError("input sequence exceeds configured context_length")

        x = self.embeddings(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        logits = self.output_projection(x)

        loss = None
        if target_ids is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                target_ids.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
        repetition_window: int | None = None,
        on_token: Callable[[int], None] | None = None,
    ) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when set")
        if repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be at least 1.0")
        if repetition_window is not None and repetition_window < 0:
            raise ValueError("repetition_window must be non-negative when set")

        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.context_length :]
            logits, _ = self(context)
            next_token_logits = logits[:, -1, :]
            next_token_logits = _apply_repetition_penalty(
                logits=next_token_logits,
                input_ids=input_ids,
                penalty=repetition_penalty,
                window=repetition_window,
            )
            next_token_logits = next_token_logits / temperature
            if top_k is not None:
                k = min(top_k, next_token_logits.shape[-1])
                top_values, top_indices = torch.topk(next_token_logits, k=k, dim=-1)
                filtered_logits = torch.full_like(next_token_logits, float("-inf"))
                next_token_logits = filtered_logits.scatter(
                    dim=-1,
                    index=top_indices,
                    src=top_values,
                )
            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            if on_token is not None:
                for token_id in next_token[:, 0].tolist():
                    on_token(int(token_id))
            input_ids = torch.cat((input_ids, next_token), dim=1)
        if was_training:
            self.train()
        return input_ids


def _apply_repetition_penalty(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float,
    window: int | None,
) -> torch.Tensor:
    if penalty == 1.0 or window == 0:
        return logits

    penalized_logits = logits.clone()
    recent_input_ids = input_ids if window is None else input_ids[:, -window:]
    for batch_index in range(recent_input_ids.shape[0]):
        repeated_token_ids = recent_input_ids[batch_index].unique()
        repeated_logits = penalized_logits[batch_index, repeated_token_ids]
        penalized_logits[batch_index, repeated_token_ids] = torch.where(
            repeated_logits < 0,
            repeated_logits * penalty,
            repeated_logits / penalty,
        )
    return penalized_logits
