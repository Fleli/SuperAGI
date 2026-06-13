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
        self.qkv_proj = nn.Linear(dim_embedding, 3 * dim_embedding)
        self.out_proj = nn.Linear(dim_embedding, dim_embedding)
        self.dropout_probability = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, dim_embedding = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

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
