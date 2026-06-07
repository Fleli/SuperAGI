import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim_embedding: int,
        n_heads: int,
        context_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if dim_embedding % n_heads != 0:
            raise ValueError("dim_embedding must be divisible by n_heads")

        self.n_heads = n_heads
        self.head_dim = dim_embedding // n_heads
        self.q_proj = nn.Linear(dim_embedding, dim_embedding)
        self.k_proj = nn.Linear(dim_embedding, dim_embedding)
        self.v_proj = nn.Linear(dim_embedding, dim_embedding)
        self.out_proj = nn.Linear(dim_embedding, dim_embedding)
        self.dropout_probability = dropout
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_length, context_length, dtype=torch.bool)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, dim_embedding = x.shape

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_probability if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            dim_embedding,
        )
        return self.out_proj(out)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, dim_embedding = x.shape
        x = x.view(batch_size, sequence_length, self.n_heads, self.head_dim)
        return x.transpose(1, 2)
