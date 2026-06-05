import torch
from torch import nn


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, dim_embedding: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim_embedding)
        self.position_embedding = nn.Embedding(context_length, dim_embedding)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, sequence_length = input_ids.shape
        position_ids = torch.arange(
            sequence_length,
            device=input_ids.device,
            dtype=torch.long,
        )
        return self.token_embedding(input_ids) + self.position_embedding(position_ids)
